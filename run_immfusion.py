"""
Training and evaluation codes for 
3D human body mesh reconstruction from images and point clouds
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
from openpyxl import load_workbook, Workbook
from src.datasets.utils import copy2cpu, convert_square_image, crop_image, project_pcl

from src.modeling.bert import BertConfig, Graphormer
import src.modeling.bert.e2e_body_network as Models
from src.modeling._smpl import SMPL, Mesh, SMPLXModel, SMPLX_MODEL_NEUTRAL_PATH
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection
import src.datasets.human_mesh_tsv as Datasets
from src.utils.loss import LossManager
from src.utils.visualization import EvaluateStreamPlot, O3DStreamPlot, pcl2box, pcl2sphere
from azureml.core.run import Run
from src.utils.dingtalk import TimerBot

aml_run = Run.get_context()

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint',
        'checkpoint-{}-{}'.format(epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))

def save_scores(args, split, mpjpe, pampjpe, mpve):
    eval_log = []
    res = {}
    res['mPJPE'] = mpjpe
    res['PAmPJPE'] = pampjpe
    res['mPVE'] = mpve
    eval_log.append(res)
    with open(op.join(args.output_dir, split+'_eval_logs.json'), 'w') as f:
        json.dump(eval_log, f)
    logger.info("Save eval scores to {}".format(args.output_dir))
    return

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        # gt_pelvis = gt[:,0,:]
        # gt = gt - gt_pelvis[:, None, :]
        # pred_pelvis = pred[:,0,:]
        # pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    # conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d)).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    # conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        # gt_pelvis = gt_keypoints_3d[:,0,:]
        # gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        # pred_pelvis = pred_pelvis[:,0,:]
        # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 
    
def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def run(args, train_dataloader, val_dataloader, Graphormer_model, smpl, mesh_sampler, renderer, start_epoch):
    smpl.eval()
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    start_iter = start_epoch * iters_per_epoch
    if iters_per_epoch < 1000:
        args.logging_steps = 500

    optimizer = torch.optim.Adam(params=list(Graphormer_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)

    # define loss function (criterion) and optimizer
    # criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        logger.info(
                ' '.join(
                ['Local rank: {o}', 'Max iteration: {a}', 'iters_per_epoch: {b}','num_train_epochs: {c}',]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    ding_bot = TimerBot()
    loss_manager = LossManager(ding_bot)

    start_training_time = time.time()
    end = time.time()
    Graphormer_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()
    log_eval_metrics = EvalMetricsLogger()

    for iteration, (pcls, images, annotations) in enumerate(train_dataloader):
        # gc.collect()
        # torch.cuda.empty_cache()
        Graphormer_model.train()
        iteration += start_iter + 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.cuda(args.device)
        pcls = pcls.cuda(args.device)
        gt_2d_joints = annotations['joints_2d'].cuda(args.device)
        # gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:]
        has_2d_joints = annotations['has_2d_joints'].cuda(args.device)

        gt_3d_joints = annotations['joints_3d'].cuda(args.device)
        gt_3d_pelvis = annotations['root_pelvis'].cuda(args.device)[:,None,:]
        # gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:]
        gt_3d_joints = gt_3d_joints - gt_3d_pelvis
        has_3d_joints = annotations['has_3d_joints'].cuda(args.device)

        gt_pose = annotations['pose'].cuda(args.device)
        gt_betas = annotations['betas'].cuda(args.device)
        has_smpl = annotations['has_smpl'].cuda(args.device)
        mjm_mask = annotations['mjm_mask'].cuda(args.device)
        mvm_mask = annotations['mvm_mask'].cuda(args.device)
        grid_mask = annotations['grid_mask'].cuda(args.device)
        cluster_mask = annotations['cluster_mask'].cuda(args.device)

        # generate simplified mesh
        gt_vertices = smpl(gt_pose, gt_betas)['verts']
        gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices, n1=0, n2=2)
        gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

        # normalize gt based on smpl's pelvis 
        gt_vertices_sub2 = gt_vertices_sub2 - gt_3d_pelvis
        gt_vertices_sub = gt_vertices_sub - gt_3d_pelvis 
        gt_vertices = gt_vertices - gt_3d_pelvis

        pcls[:,:,:3] -= gt_3d_pelvis

        # prepare masks for mask vertex/joint modeling
        mjm_mask_ = mjm_mask.expand(-1,-1,2051)
        mvm_mask_ = mvm_mask.expand(-1,-1,2051)
        grid_mask_ = grid_mask.expand(-1,-1,2051)
        cluster_mask_ = cluster_mask.expand(-1,-1,2051)
        meta_masks = torch.cat([mjm_mask_, mvm_mask_, grid_mask_, cluster_mask_], dim=1)

        # forward-pass
        pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = Graphormer_model(args, images, smpl, mesh_sampler, meta_masks=meta_masks, is_train=True, pcls=pcls)

        # obtain 3d joints, which are regressed from the full mesh
        # pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
        # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]

        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        # pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
        # pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

        # compute 3d joint loss  (where the joints are directly output from transformer)
        loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints, args.device)
        # compute 3d vertex loss
        loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2, has_smpl, args.device) + \
                            args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub, has_smpl, args.device) + \
                            args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, args.device))
        # compute 3d joint loss (where the joints are regressed from full mesh)
        # loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints, has_3d_joints, args.device)
        # compute 2d joint loss
        # loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints)
                        #  keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints, has_2d_joints)
        score_loss = torch.mean(torch.abs(pred_dict.get('pred_score', torch.zeros(1, device=args.device))))
        image_joints_loss = keypoint_3d_loss(criterion_keypoints, pred_dict.get('pred_image_joints', gt_3d_joints), gt_3d_joints, has_3d_joints, args.device)
        point_joints_loss = keypoint_3d_loss(criterion_keypoints, pred_dict.get('pred_point_joints', gt_3d_joints), gt_3d_joints, has_3d_joints, args.device)
        image_verts_loss = vertices_loss(criterion_vertices, pred_dict.get('pred_image_verts', gt_vertices_sub2), gt_vertices_sub2, has_smpl, args.device)
        point_verts_loss = vertices_loss(criterion_vertices, pred_dict.get('pred_point_verts', gt_vertices_sub2), gt_vertices_sub2, has_smpl, args.device)
        
        # loss_manager.update_loss("2d_joint_loss", args.joints2d_loss_weight*loss_2d_joints)

        loss_manager.update_loss("score_loss", args.score_loss_weight*score_loss)
        loss_manager.update_loss("image_joints_loss", args.joints_loss_weight*image_joints_loss)
        loss_manager.update_loss("point_joints_loss", args.joints_loss_weight*point_joints_loss)
        loss_manager.update_loss("image_verts_loss", args.vertices_loss_weight*image_verts_loss)
        loss_manager.update_loss("point_verts_loss", args.vertices_loss_weight*point_verts_loss)
        
        loss_manager.update_loss("3d_joint_loss", args.joints_loss_weight*loss_3d_joints)
        loss_manager.update_loss("vertex_loss", args.vertices_loss_weight*loss_vertices)
        
        # we empirically use hyperparameters to balance difference losses
        loss = loss_manager.calculate_total_loss()

        # update logs
        # log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        log_losses.update(loss.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2djoints.avg, log_loss_3djoints.avg, log_loss_vertices.avg, batch_time.avg, data_time.avg, 
                    optimizer.param_groups[0]['lr'])
            )

            aml_run.log(name='Loss', value=float(log_losses.avg))
            aml_run.log(name='3d joint Loss', value=float(log_loss_3djoints.avg))
            aml_run.log(name='2d joint Loss', value=float(log_loss_2djoints.avg))
            aml_run.log(name='vertex Loss', value=float(log_loss_vertices.avg))

        if iteration % iters_per_epoch == 0:
            loss_manager.calculate_epoch_loss(os.path.join(args.output_dir, 'loss/train'), epoch)
            val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                                Graphormer_model, 
                                                criterion_keypoints, 
                                                criterion_vertices, 
                                                epoch, 
                                                smpl,
                                                mesh_sampler,
                                                renderer,
                                                loss_manager)
            
            aml_run.log(name='mPVE', value=float(1000*val_mPVE))
            aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
            aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))
            logger.info(
                ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
                + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, Data Count: {:6.2f}'.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE, val_count)
            )

            if val_PAmPJPE<log_eval_metrics.PAmPJPE or epoch%10==0:
                save_checkpoint(Graphormer_model, args, epoch, iteration)
                log_eval_metrics.update(val_mPVE, val_mPJPE, val_PAmPJPE, epoch)
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    save_checkpoint(Graphormer_model, args, epoch, iteration)

    logger.info(
        ' Best Results:'
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, at epoch {:6.2f}'.format(1000*log_eval_metrics.mPVE, 1000*log_eval_metrics.mPJPE, 1000*log_eval_metrics.PAmPJPE, log_eval_metrics.epoch)
    )


def run_eval_general(args, val_dataloader, Graphormer_model, smpl, mesh_sampler, renderer):
    smpl.eval()
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    epoch = 0
    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    Graphormer_model.eval()

    val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                    Graphormer_model, 
                                    criterion_keypoints, 
                                    criterion_vertices, 
                                    epoch, 
                                    smpl,
                                    mesh_sampler,
                                    renderer)

    aml_run.log(name='mPVE', value=float(1000*val_mPVE))
    aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
    aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))

    logger.info(
        ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f} '.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE)
    )
    return

def res_generator(pcls, pred_vertices, gt_vertices, face, pred_verts_sub2, pred_joints, attention, seed_points, joint_id, images, gt_pelvis, show_att=False, pause_at_start=False):
    for i in range(len(pcls)):
        print('frame_id:', i)
        res = dict(
            radar_mesh = dict(
                mesh = pcl2box(copy2cpu(pcls[i])),
                color = [0,0.8,0],
            ),
            pred_smpl = dict(
                mesh = [copy2cpu(pred_vertices[i]), copy2cpu(face)],
                color = np.asarray([249, 239, 211]) /255
                # 179, 230, 213
            ),
            label_smpl = dict(
                mesh = [copy2cpu(gt_vertices[i]), copy2cpu(face)],
                color = np.asarray([235, 189, 191]) / 255,
            ),
        )
        if show_att:
            res = show_att_weight(copy2cpu(pred_verts_sub2[i]), copy2cpu(pred_joints[i]), copy2cpu(attention[i]), copy2cpu(seed_points[i]), joint_id, copy2cpu(images[i]), copy2cpu(pcls[i]), copy2cpu(gt_pelvis[i]))
        O3DStreamPlot.pause = pause_at_start
        yield res

def run_validate(args, val_loader, Graphormer_model, criterion, criterion_vertices, epoch, smpl, mesh_sampler, renderer=None, loss_manager=None):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()

    per_joint_err = []
    per_vertex_err = []

    if not args.train:
        snapshot_path = os.path.join(args.output_dir, 'snapshot', args.test_scene)
        if not os.path.isdir(snapshot_path):
            os.makedirs(snapshot_path)
        plot = EvaluateStreamPlot(save_path=snapshot_path)
    # switch to evaluate mode
    Graphormer_model.eval()
    smpl.eval()
    batch = 0
    with torch.no_grad():
        # end = time.time()
        for i, (pcls, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            # compute output
            images = images.cuda(args.device)
            pcls = pcls.cuda(args.device)
            gt_3d_joints = annotations['joints_3d'].cuda(args.device)
            gt_3d_pelvis = annotations['root_pelvis'].cuda(args.device)[:,None,:]
            # gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] 
            gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis
            has_3d_joints = annotations['has_3d_joints'].cuda(args.device)

            gt_pose = annotations['pose'].cuda(args.device)
            gt_betas = annotations['betas'].cuda(args.device)
            has_smpl = annotations['has_smpl'].cuda(args.device)

            # generate simplified mesh
            mesh_res = smpl(gt_pose, gt_betas)
            gt_vertices = mesh_res['verts']
            gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
            gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices_sub, n1=1, n2=2)

            # normalize gt based on smpl pelvis 
            gt_vertices_sub2 = gt_vertices_sub2 - gt_3d_pelvis
            gt_vertices_sub = gt_vertices_sub - gt_3d_pelvis 
            gt_vertices = gt_vertices - gt_3d_pelvis
            pcls[:,:,:3] -= gt_3d_pelvis

            # forward-pass
            pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = Graphormer_model(args, images, smpl, mesh_sampler, pcls=pcls)

            # obtain 3d joints from full mesh
            # pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

            # pred_3d_pelvis = pred_3d_joints[:, 0, :]
            # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            # pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            # measure errors
            error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
            error_joints = mean_per_joint_position_error(pred_3d_joints, gt_3d_joints, has_3d_joints)
            error_joints_pa = reconstruction_error(pred_3d_joints.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)

            per_joint_err.append(torch.norm((pred_3d_joints - gt_3d_joints), dim=-1))
            per_vertex_err.append(torch.norm((pred_vertices - gt_vertices), dim=-1))

            if len(error_vertices)>0:
                mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )
            if len(error_joints)>0:
                mPJPE.update(np.mean(error_joints), int(torch.sum(has_3d_joints)) )
            if len(error_joints_pa)>0:
                PAmPJPE.update(np.mean(error_joints_pa), int(torch.sum(has_3d_joints)) )

            if loss_manager is not None:
                loss_manager.update_loss("mPVE", np.mean(error_vertices))
                loss_manager.update_loss("mPJPE", np.mean(error_joints))
                loss_manager.update_loss("PAmPJPE", np.mean(error_joints_pa))
                loss_manager.calculate_total_loss()

            if args.visual:
                att = pred_dict.get('attention', [0])[-1]
                seed_points = pred_dict.get('seed_points', None)
                gen = res_generator(pcls[:,:,:3], pred_vertices, gt_vertices, smpl.f, pred_vertices_sub2, pred_3d_joints, 
                                    att, seed_points, args.joint_id, annotations['orig_img'], gt_3d_pelvis, args.show_att, pause_at_start=True)
                batch += 1
                if batch > args.max_num_batch:
                    return 0, 0, 0, 0
                if args.save_snapshot:
                    print('batch_id:', i)
                    plot.show(gen, save_path=snapshot_path)
                else:
                    plot.show(gen)

    if loss_manager is not None:
        loss_manager.calculate_epoch_loss(os.path.join(args.output_dir, 'loss/eval'), epoch)
    
    val_mPVE = all_gather(float(mPVE.avg))
    val_mPVE = sum(val_mPVE)/len(val_mPVE)
    val_mPJPE = all_gather(float(mPJPE.avg))
    val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)

    val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)

    val_count = all_gather(float(mPVE.count))
    val_count = sum(val_count)

    # save error
    if not args.train:
        output_path = os.path.join(args.output_dir, 'error', args.test_scene)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        j_err = copy2cpu(torch.vstack(per_joint_err))
        v_err = copy2cpu(torch.vstack(per_vertex_err))
        np.save(os.path.join(output_path, "per_joint_err"), j_err)
        np.save(os.path.join(output_path, "per_vertex_err"), v_err)
        print("mean joint err (cm):", np.mean(j_err)*100)
        print("mean vertex err (cm):", np.mean(v_err)*100)
        print("max joint err (cm):", np.mean(np.max(j_err, axis=1), axis=0)*100)
        print("max vertex err (cm):", np.mean(np.max(v_err, axis=1), axis=0)*100)
        with open(os.path.join(output_path, "error.txt"), 'w') as f:
            f.write("mean joint error: " + str(np.mean(j_err)*100))
            f.write("\nmean vertex error: " + str(np.mean(v_err)*100))
            f.write("\nmax joint error: " + str(np.mean(np.max(j_err, axis=1), axis=0)*100))
            f.write("\nmax vertex error: " + str(np.mean(np.max(v_err, axis=1), axis=0)*100))

        sheet_loc_dict = {
            'lab1': ['A1', 'B1', 'A2', 'B2'],
            'lab2': ['C1', 'D1', 'C2', 'D2'],
            'furnished': ['E1', 'F1', 'E2', 'F2'],
            'rain': ['G1', 'H1', 'G2', 'H2'],
            'smoke': ['I1', 'J1', 'I2', 'J2'],
            'poor_lighting': ['K1', 'L1', 'K2', 'L2'],
            'occlusion': ['M1', 'N1', 'M2', 'N2'],
        }
        sheet_locs = sheet_loc_dict[args.test_scene]
        excel_path = os.path.join(args.output_dir, 'error', 'error.xlsx')
        if not os.path.exists(excel_path):
            work_book = Workbook()
            work_sheet = work_book.active
            work_sheet.title = "Sheet1"
        else:
            work_book = load_workbook(filename=excel_path)
            work_sheet = work_book['Sheet1']
        work_sheet[sheet_locs[0]] = str(np.mean(j_err)*100)
        work_sheet[sheet_locs[1]] = str(np.mean(v_err)*100)
        work_sheet[sheet_locs[2]] = str(np.mean(np.max(j_err, axis=1), axis=0)*100)
        work_sheet[sheet_locs[3]] = str(np.mean(np.max(v_err, axis=1), axis=0)*100)
        work_book.save(excel_path)

    return val_mPVE, val_mPJPE, val_PAmPJPE, val_count


def visualize_mesh( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def visualize_mesh_test( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d,
                    PAmPJPE_h36m_j14):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        score = PAmPJPE_h36m_j14[i]
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, score)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

    
def show_att_weight(vertices_sub2, joints_3d, attention, seed_points, joint_id, image, pcl, gt_pelvis):
    heads_num, vertex_num, _ = attention.shape
    all_head = np.zeros((vertex_num,vertex_num))

    # compute avg attention for 4 attention heads
    for h in range(heads_num):
        att_per_img = attention[h]
        all_head = all_head + att_per_img   
    all_head = all_head/4

    col_sums = all_head.sum(axis=0)
    all_head = all_head / col_sums[np.newaxis, :]

    # select the joint to show attention
    ref_joint = joints_3d[joint_id]
    attention_to_show = all_head[joint_id][22:758:]
    # nomalize attention
    min_v = np.min(attention_to_show)
    max_v = np.max(attention_to_show)
    norm_attention_to_show = (attention_to_show - min_v)/(max_v-min_v)

    # get verts attention lines
    verts_to_show = [ref_joint,]
    verts_lines = []
    verts_line_colors = []
    thres = 0.1
    for i in range(vertices_sub2.shape[0]):
        if norm_attention_to_show[i] > thres:
            verts_to_show.append(vertices_sub2[i])
            verts_lines.append([0, len(verts_to_show)-1])
            verts_line_colors.append(1 + thres - norm_attention_to_show[i] * np.asarray((200, 208, 255)) / 255)
    
    # # init grid boxes
    # grid_box = np.full((7, 7, 3), [1.2, 0., 0.])
    # for i in range(7):
    #     grid_box[i,:,1] += 0.1 * i
    #     for j in range(7):
    #         grid_box[i,j,0] += 0.1 * j
    # grid_box = grid_box.reshape(-1, 3)
    
    # init cluster sphere
    # cluster_sphere = seed_points + np.array([-2., 0., 0.])
    
    # # get feature attention lines
    # feat_centers = np.vstack((grid_box, cluster_sphere))
    # feat_to_show = [ref_joint,]
    # feat_lines = []
    # feat_line_colors = []
    # for i in range(len(feat_centers)):
    #     if norm_attention_to_show[i + 655] > thres:
    #         feat_to_show.append(feat_centers[i])
    #         feat_lines.append([0, len(feat_to_show)-1])
    #         feat_line_colors.append(1 + thres - norm_attention_to_show[i + 655] * np.asarray([235, 189, 191]) / 255)
    
    # get attention map
    pcl_2d = project_pcl(pcl + gt_pelvis)
    img_with_cluster = image.copy()
    center_2d = project_pcl(seed_points + gt_pelvis)
    blank = np.zeros_like(img_with_cluster[:,:,:1])
    weight_sorted_idx = np.argsort(norm_attention_to_show[[i+704 for i in range(center_2d.shape[0])]])
    for i in weight_sorted_idx:
        # cv2.circle(blank, (center_2d[i, 1], center_2d[i, 0]), 60, norm_attention_to_show[i+704]*np.array([255,255,255]), -1)
        cv2.circle(blank, (center_2d[i, 1], center_2d[i, 0]), 60, norm_attention_to_show[i+704]*255, -1)
    blank = cv2.applyColorMap(blank, 2)
    img_with_cluster = cv2.addWeighted(img_with_cluster, 0.2, blank, 0.5, 1)
    
    img_with_cluster[pcl_2d[:, 0], pcl_2d[:, 1]] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0]-1, pcl_2d[:, 1]-1] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0]-1, pcl_2d[:, 1]] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0], pcl_2d[:, 1]-1] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0]+1, pcl_2d[:, 1]+1] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0]+1, pcl_2d[:, 1]] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0], pcl_2d[:, 1]+1] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0]-1, pcl_2d[:, 1]+1] = [0, 255, 0]
    img_with_cluster[pcl_2d[:, 0]+1, pcl_2d[:, 1]-1] = [0, 255, 0]
    img_with_cluster = crop_image(joints_3d+gt_pelvis, img_with_cluster, margin=0.3, square=True)
    img_with_cluster = cv2.resize(img_with_cluster, (224, 224))
    
    image = crop_image(joints_3d+gt_pelvis, image, margin=0.3, square=True)
    image = cv2.resize(image, (224, 224))
    box_width = image.shape[0]//7
    blank = np.zeros_like(image[:,:,:1])
    for i in range(7):
        for j in range(7):
            cv2.rectangle(blank, (i*box_width,j*box_width), ((i+1)*box_width,(j+1)*box_width), norm_attention_to_show[i*7+j+655]*255, -1)
    blank = cv2.applyColorMap(blank, 2)
    img_with_grid = cv2.addWeighted(image, 0.5, blank, 0.5, 1)
    heatmap = np.hstack((img_with_cluster, img_with_grid))
    cv2.imwrite('heatmap.png', heatmap)
    
    cv2.namedWindow('heatmap', 0)
    cv2.resizeWindow("heatmap", 640, 480)
    cv2.imshow('heatmap', heatmap)
    cv2.waitKey(1)

    res = dict(
        vert_pcl=dict(pcl=vertices_sub2, color=np.asarray([255, 181, 74])/255,),
        vert_lines=dict(pcl=verts_to_show, lines=np.asarray(verts_lines), colors=verts_line_colors),
        # grid_box=dict(mesh=pcl2box(grid_box, 0.04), color=np.asarray([208, 163, 230])/255,),
        # cluster_sphere=dict(mesh=pcl2sphere(cluster_sphere, 0.1), color=np.asarray([179, 230, 213])/255,),
        # feat_lines=dict(pcl=feat_to_show, lines=np.asarray(feat_lines), colors=feat_line_colors),
        # radar_pcl = dict(pcl = pcl + np.array([-2., 0., 0.]), color = [0,0.8,0],),
    )

    return res


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    args.distributed = args.num_gpus > 1
    torch.cuda.set_device(args.gpu_idx)
    args.device = torch.device(args.device)
    if args.distributed:
        print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0, device=args.device).to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=smpl.f.cpu().numpy())

    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [args.output_dim]

    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]
    
    if args.train==False and args.resume_checkpoint!=None and args.resume_checkpoint!='None':
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(os.path.join(args.resume_checkpoint, 'model.bin'))
    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.device = args.device
            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*args.interm_size_scale)

            if which_blk_graph[i]==1:
                config.graph_conv = True
                logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = args.mesh_type

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        
        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])


        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer graphormer encoder)
        Model = getattr(Models, args.model)
        _model = Model(args, config, backbone, trans_encoder, mesh_sampler)

        start_epoch = 0
        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            start_epoch = int(args.resume_checkpoint.split('-')[1])
            # workaround approach to load sparse tensor in graph conv.
            states = torch.load(os.path.join(args.resume_checkpoint, 'state_dict.bin'))
            # states = checkpoint_loaded.state_dict()
            for k, v in states.items():
                states[k] = v.cpu()
            # del checkpoint_loaded
            _model.load_state_dict(states, strict=False)
            del states
            gc.collect()
            torch.cuda.empty_cache()

    # update configs to enable attention outputs
    if args.show_att:
        setattr(_model.trans_encoder[-1].config,'output_attentions', True)
        setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
        _model.trans_encoder[-1].bert.encoder.output_attentions = True
        _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
        for iter_layer in range(4):
            _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
        setattr(_model.trans_encoder[-1].config,'device', args.device)

    _model.to(args.device)
    logger.info("Training parameters %s", args)
    seq_idxes = eval(args.seq_idxes) if args.seq_idxes else range(20)
    Dataset = getattr(Datasets, args.dataset)
    dataset = Dataset(
        args.data_path, 
        args.train, 
        args.test_scene, 
        use_pkl=args.use_pkl, 
        create_pkl=args.create_pkl, 
        seq_idxes=seq_idxes,
        num_points=args.num_points,
        add_rgb=args.add_rgb,
        skip_head=args.skip_head,
        skip_tail=args.skip_tail,
    )

    if args.train==True:
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        dataset_train, dataset_eval = torch.utils.data.random_split(dataset, [train_size, eval_size])
        train_dataloader = make_data_loader(args, dataset_train, args.distributed, is_train=True, 
                                            scale_factor=args.img_scale_factor)
        val_dataloader = make_data_loader(args, dataset_eval, args.distributed, is_train=False, 
                                            scale_factor=args.img_scale_factor)
        run(args, train_dataloader, val_dataloader, _model, smpl, mesh_sampler, renderer, start_epoch)
        
    else:
        val_dataloader = make_data_loader(args, dataset, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_general(args, val_dataloader, _model, smpl, mesh_sampler, renderer)


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_path", default='/home/nesc525/drivers/0/dataset', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='Tax-H36m-coco40k-Muco-UP-Mpii/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=30, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=50, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--joints2d_loss_weight", default=0, type=float)
    parser.add_argument("--score_loss_weight", default=100.0, type=float)
    parser.add_argument("--vloss_w_full", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float) 
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--model", default='mmGraphormer', type=str,
                        help='Choose the model')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--which_gcn", default='0,0,1', type=str, 
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv") 
    parser.add_argument("--mesh_type", default='body', type=str, help="body or hand") 
    parser.add_argument("--interm_size_scale", default=2, type=int)
    #########################################################
    # Others
    #########################################################
    parser.add_argument('--train', dest="train", action="store_true", help='train or test')
    parser.add_argument("--test_scene", type=str, default='lab1')
    parser.add_argument("--seq_idxes", type=str, default='') 
    parser.add_argument('--visual', dest="visual", action="store_true", help='visual')
    parser.add_argument('--logging_steps', type=int, default=1000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument("--gpu_idx", type=int, default=0, )
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")
    parser.add_argument('--create_pkl', dest="create_pkl", action="store_true", help='create pkl data')
    parser.add_argument('--use_pkl', dest="use_pkl", action="store_true", help='use pkl data')
    parser.add_argument('--skip_head', type=int, default=0)
    parser.add_argument('--skip_tail', type=int, default=0)
    parser.add_argument('--max_num_batch', type=int, default=10000)
    parser.add_argument('--save_snapshot', action="store_true")
    parser.add_argument('--output_dim', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='mmBodyDataset')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--add_rgb', action="store_true", help='add rgb values')
    parser.add_argument('--points_w_image_feat', action="store_true")
    parser.add_argument('--wo_GIM', action="store_true")
    parser.add_argument('--images_only', action="store_true")
    parser.add_argument('--points_only', action="store_true")
    parser.add_argument('--cat_global_feat', action="store_true")
    parser.add_argument('--only_grid_feat', action="store_true")
    parser.add_argument('--only_cluster_feat', action="store_true")
    parser.add_argument('--show_att', action="store_true")
    parser.add_argument('--joint_id', type=int, default=0)
    parser.add_argument('--max_position_embeddings', type=int, default=800)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
