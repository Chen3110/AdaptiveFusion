import copy
import datetime
import os
import time
import sys
import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
import torch
import torch.utils.data
from torch import nn
import torchvision
from src.modeling._smpl import SMPLX
from src.datasets.utils import copy2cpu as c2c
from src.datasets.utils import rotation6d_2_rot_mat, rodrigues_2_rot_mat
from src.modeling.loss import GeodesicLoss
from src.utils.geometric_layers import orthographic_projection
import src.utils.comm as utils
import src.datasets.fusion_dataset as Datasets
from src.utils.logger import setup_logger
from src.utils.scheduler import WarmupMultiStepLR
import src.modeling.model as Models
from torch.utils.tensorboard import SummaryWriter


def result_generator(args, data_dict, smpl_results, face):
    for i in range(len(data_dict['joints_3d'])):
        yield dict(
            radar0 = dict(
                # mesh = pcl2box(data_dict['radar'][i][:,:3]) if data_dict['radar'].shape[1] else None,
                pcl = c2c(data_dict['radar0'][i][:,:3]) if 'radar0' in args.inputs else None,
                color = [0,0.8,0],
            ),
            depth0 = dict(
                # mesh = pcl2box(data_dict['master_depth'][i][:,:3]) if data_dict['master_depth'].shape[1] else None,
                pcl = c2c(data_dict['depth0'][i][:,:3]) if 'depth0' in args.inputs else None,
                color = np.asarray([59, 170, 235]) /255,
            ),
            depth1 = dict(
                # mesh = pcl2box(data_dict['sub_depth'][i][:,:3]) if data_dict['sub_depth'].shape[1] else None,
                pcl = c2c(data_dict['depth1'][i][:,:3]) if 'depth1' in args.inputs else None,
                color = np.asarray([245, 157, 86]) /255,
            ),
            pred_smpl = dict(
                mesh = [c2c(smpl_results['vertices'][i]), c2c(face)],
                color = np.asarray([208, 163, 230]) /255,
            ),
            label_smpl = dict(
                mesh = [c2c(data_dict['vertices'][i]), c2c(face)],
                color = np.asarray([235, 189, 191]) / 255,
            )
        )

def train_one_epoch(args, model, body_model, loss_manager, criterions, loss_weight, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('batch/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for iteration, data_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        if isinstance(data_dict, dict):
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(args.device, dtype=torch.float32)
        else:
            data_dict = data_dict.to(device, dtype=torch.float32)
        smpl_params_gt = data_dict['smpl_params'][:,3:].to(device, dtype=torch.float32)
        output_dict = model(data_dict, is_train=True)
        pred_smpl_params = output_dict['pred_smpl_params']
        batch_size = smpl_params_gt.shape[0]
        # translation
        trans_gt = data_dict['smpl_params'][:,:3].to(device, dtype=torch.float32)
        trans_loss = 0
        if args.trans_loss:
            trans_loss = loss_weight[0]*criterions["mse"](output_dict['pred_trans'], trans_gt)
            loss_manager.add_scalar("trans_loss", trans_loss.item(), iteration)
        # pose loss
        if args.use_6d_pose:
            output_mat = rotation6d_2_rot_mat(pred_smpl_params[:,:-16])
            target_mat = rodrigues_2_rot_mat(smpl_params_gt[:,:-16])
            pose_loss = loss_weight[1]*criterions["rot_mat"](output_mat, target_mat)
            smpl_results = body_model(torch.cat((trans_gt, output_mat),-1), pred_smpl_params[:,-16:], use_rodrigues=False)
        else:
            pose_loss = loss_weight[1]*criterions["mse"](pred_smpl_params[:,:-16],smpl_params_gt[:,:-16])
            smpl_results = body_model(torch.cat((trans_gt, output_mat), -1), pred_smpl_params[:,-16:])
        loss_manager.add_scalar("pose_loss", pose_loss.item(), iteration)
        # shape loss
        shape_loss = loss_weight[2]*criterions["mse"](pred_smpl_params[:,-16:], smpl_params_gt[:,-16:])
        loss_manager.add_scalar("shape_loss", shape_loss.item(), iteration)
        # joints loss
        joints_loss = loss_weight[3]*criterions["l1"](smpl_results['joints'][:,:22], data_dict['joints_3d'])
        loss_manager.add_scalar("joints_loss", joints_loss.item(), iteration)
        # vertices loss
        vertices_loss = loss_weight[4]*criterions["l1"](smpl_results['vertices'], data_dict['vertices'])
        loss_manager.add_scalar("vertices_loss", vertices_loss.item(), iteration)
        # gender loss
        gender_loss = 0
        if args.use_gender:
            gender_loss = loss_weight[5]*criterions["entropy"](pred_smpl_params[:,-1], smpl_params_gt[:,-1])
            loss_manager.add_scalar("gender_loss", gender_loss.item(), iteration)
        # 2d joints loss
        joints_2d_loss = 0
        if args.joints_2d_loss:
            pred_2d_joints = orthographic_projection(smpl_results['joints'][:,:22] - trans_gt[:,None], output_dict['pred_cam'])
            joints_2d_loss = loss_weight[3]*criterions["mse"](pred_2d_joints, data_dict['joints_2d'])
            loss_manager.add_scalar("2d_joint_loss", joints_2d_loss.item(), iteration)

        loss = trans_loss + pose_loss + shape_loss + joints_loss + vertices_loss + gender_loss + joints_2d_loss
        optimizer.zero_grad()
        #with torch.autograd.detect_anomaly():
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['batch/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()

def evaluate(args, model, body_model, loss_manager, criterions, data_loader, device, save_path=''):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", logger=logger)
    header = 'Test:'
    per_joint_err = []
    per_vertex_err = []
    shape_err = []
    batch = 0
    snapshot_path = os.path.join(args.output_dir, 'snapshot', args.test_scene)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if args.visual:
        from src.utils.visualization import EvaluateStreamPlot
        plot = EvaluateStreamPlot(save_path=snapshot_path)
    else:
        from src.utils.renderer import Renderer, visualize_mesh
        renderer = Renderer(faces=body_model.faces.cpu().numpy())

    with torch.no_grad():
        for iteration, data_dict in enumerate(metric_logger.log_every(data_loader, 100, header)):
            if isinstance(data_dict, dict):
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict[k] = v.to(device, non_blocking=True, dtype=torch.float32)
            else:
                data_dict = data_dict.to(device, non_blocking=True, dtype=torch.float32)
            smpl_params_gt = data_dict['smpl_params'][:,3:].to(device, non_blocking=True, dtype=torch.float32)
            batch_size = smpl_params_gt.shape[0]
            output_dict = model(data_dict, is_train=False)
            pred_smpl_params = output_dict['pred_smpl_params']
            # translation
            trans_gt = data_dict['smpl_params'][:,:3].to(device, non_blocking=True, dtype=torch.float32)
            trans_loss = 0
            if args.trans_loss:
                trans_loss = criterions["mse"](output_dict['pred_trans'], trans_gt)
                loss_manager.add_scalar("trans_loss", trans_loss.item(), iteration)
            # pose loss
            if args.use_6d_pose:
                output_mat = rotation6d_2_rot_mat(pred_smpl_params[:,:-16])
                target_mat = rodrigues_2_rot_mat(smpl_params_gt[:,:-16])
                pose_loss = criterions["rot_mat"](output_mat, target_mat)
                smpl_results = body_model(torch.cat((trans_gt, output_mat),-1), pred_smpl_params[:,-16:], use_rodrigues=False)
            else:
                pose_loss = criterions["mse"](pred_smpl_params[:,:-16],smpl_params_gt[:,:-16])
                smpl_results = body_model(torch.cat((trans_gt, pred_smpl_params[:,:-16]), -1), pred_smpl_params[:,-16:])
            loss_manager.add_scalar("pose_loss", pose_loss.item(), iteration)
            # shape loss
            shape_loss = criterions["mse"](pred_smpl_params[:,-16:], smpl_params_gt[:,-16:])
            loss_manager.add_scalar("shape_loss", shape_loss.item(), iteration)
            # joints loss
            joints_loss = criterions["l1"](smpl_results['joints'][:,:22], data_dict['joints_3d'])
            loss_manager.add_scalar("joints_loss", joints_loss.item(), iteration)
            # vertices loss
            vertices_loss = criterions["l1"](smpl_results['vertices'], data_dict['vertices'])
            loss_manager.add_scalar("vertices_loss", vertices_loss.item(), iteration)
            # gender loss
            gender_loss = 0
            if args.use_gender:
                gender_loss = criterions["entropy"](pred_smpl_params[:,-1], smpl_params_gt[:,-1])
                loss_manager.add_scalar("gender_loss", gender_loss.item(), iteration)
            # 2d joints loss
            joints_2d_loss = 0
            if args.joints_2d_loss:
                pred_2d_joints = orthographic_projection(smpl_results['joints'][:,:22] - trans_gt[:,None], output_dict['pred_cam'])
                joints_2d_loss = criterions["mse"](pred_2d_joints, data_dict['joints_2d'])
                loss_manager.add_scalar("joints_2d_loss", joints_2d_loss.item(), iteration)

            loss = trans_loss + pose_loss + shape_loss + joints_loss + vertices_loss + gender_loss + joints_2d_loss

            joint_err = torch.norm((smpl_results['joints'][:,:22] - data_dict['joints_3d']), dim=-1)
            vert_err = torch.norm((smpl_results['vertices'] - data_dict['vertices']), dim=-1)
            shape_err.append(abs(pred_smpl_params[:,-16:] - smpl_params_gt[:,-16:]))
            per_joint_err.append(joint_err)
            per_vertex_err.append(vert_err)

            # could have been padded in distributed setup
            batch_size = smpl_params_gt.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['loss'].update(loss, n=batch_size)

            if not args.train and args.joints_2d_loss and iteration%100==0 and not args.visual:
                pred_2d_joints = orthographic_projection(smpl_results['joints'][:,:22] - trans_gt[:,None], output_dict['pred_cam'])
                visual_imgs_output = visualize_mesh(renderer,
                                                    data_dict['orig_img'][args.joints_2d_loss].detach()/255.,
                                                    data_dict['joints_2d'][:,:,:2].detach(),
                                                    smpl_results['vertices'].detach(), 
                                                    output_dict['pred_cam'].detach(),
                                                    pred_2d_joints.detach())
                visual_imgs = visual_imgs_output.permute(1,2,0).numpy()
                temp_fname = os.path.join(args.output_dir, 'snapshot', 'pred_{}.jpg'.format(i))
                print('save to ', temp_fname)
                cv2.imwrite(temp_fname, np.asarray(visual_imgs * 255))
                
            if args.visual and not args.train:
                gen = result_generator(args, data_dict, smpl_results, body_model.f)
                batch += 1
                if batch > args.num_frames:
                    return
                if args.save_snapshot:
                    plot.show(gen, save_path=snapshot_path)
                else:
                    plot.show(gen)

        logger.info("joints error: {}".format(torch.mean(torch.cat(per_joint_err))))
        logger.info("vertices error: {}".format(torch.mean(torch.cat(per_vertex_err))))

        if save_path:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            j_err = c2c(torch.vstack(per_joint_err))
            v_err = c2c(torch.vstack(per_vertex_err))
            s_err = c2c(torch.vstack(shape_err))
            np.save(os.path.join(save_path, "per_joint_err"), j_err)
            np.save(os.path.join(save_path, "per_vertex_err"), v_err)
            np.save(os.path.join(save_path, "shape_err"), s_err)
            logger.info("mean joint err (cm): {}".format(np.mean(j_err)*100))
            logger.info("mean vertex err (cm): {}".format(np.mean(v_err)*100))
            logger.info("max joint err (cm): {}".format(np.mean(np.max(j_err, axis=1), axis=0)*100))
            logger.info("max vertex err (cm): {}".format(np.mean(np.max(v_err, axis=1), axis=0)*100))
            logger.info("mean shape err (cm): {}".format(np.mean(s_err)))
            with open(os.path.join(save_path, "error.txt"), 'w') as f:
                f.write("mean joint error: " + str(np.mean(j_err)*100))
                f.write("\nmean vertex error: " + str(np.mean(v_err)*100))
                f.write("\nmax joint error: " + str(np.mean(np.max(j_err, axis=1), axis=0)*100))
                f.write("\nmax vertex error: " + str(np.mean(np.max(v_err, axis=1), axis=0)*100))
                f.write("\nmean shape error: " + str(np.mean(s_err)))
            # write errors to the excel
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
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


def main(args):
    global logger
    output_dir = args.output_dir
    if output_dir and not os.path.exists(os.path.join(output_dir, 'checkpoint')):
        utils.mkdir(os.path.join(output_dir, 'checkpoint'))
    logger = setup_logger("AdaptiveFusion", args.output_dir, utils.get_rank())

    logger.info(args)
    logger.info("torch version: {}".format(torch.__version__))
    logger.info("torchvision version: {}".format(torchvision.__version__))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.cuda.set_device(args.gpu_idx)
    device = torch.device('cuda')
    
    args.inputs = args.inputs.replace(' ','').split(',')
    args.input_dict = {}
    for m in ['image', 'depth', 'radar']:
        args.input_dict[m] = [i for i in args.inputs if m in i]

    # Data loading code
    Dataset = getattr(Datasets, args.dataset)
    dataset = Dataset(args)

    if args.eval_test_dataset:
        dataset_train = dataset
        eval_args = copy.deepcopy(args)
        eval_args.train = False
        dataset_eval = Dataset(eval_args)
    else:
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        dataset_train, dataset_eval = torch.utils.data.random_split(dataset, [train_size, eval_size])

    logger.info("Creating data loaders")

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    logger.info("Creating model")
    Model = getattr(Models, args.model)
    model = Model(args)
    body_model = SMPLX().to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)
    
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
    loss_manager_train = SummaryWriter(args.output_dir + '/loss/train/' + TIMESTAMP)
    loss_manager_eval = SummaryWriter(args.output_dir + '/loss/eval/' + TIMESTAMP)

    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    rot_mat_criterion = GeodesicLoss()
    entropy_criterion = nn.BCEWithLogitsLoss()
    criterions = dict(mse=mse_criterion, l1=l1_criterion, rot_mat=rot_mat_criterion, entropy=entropy_criterion)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume_checkpoint:
        resume = args.resume_checkpoint
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.train:
        logger.info("Start training")
        start_time = time.time()

        loss_weight = list(map(float, args.loss_weight.split(",")))

        for epoch in range(args.start_epoch, args.epochs):
            train_one_epoch(args, model, body_model, loss_manager_train, criterions, loss_weight, optimizer, lr_scheduler, data_loader_train, device, epoch)
            evaluate(args, model, body_model, loss_manager_eval, criterions, data_loader_eval, device)

            if output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(checkpoint, os.path.join(output_dir, 'checkpoint', 'checkpoint.pth'))
                if (epoch + 5) % 5 == 0:
                    utils.save_on_master(checkpoint, os.path.join(output_dir, 'checkpoint', 'epoch{}.pth'.format(epoch)))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
    else:
        data_loader_test = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        logger.info("Start testing")
        save_path = os.path.join(output_dir, "error", args.test_scene)
        evaluate(args, model, body_model, loss_manager_eval, criterions, data_loader_test, device, save_path)
        
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='AdaptiveFusion Model Training')

    parser.add_argument('--seed', default=35, type=int, help='random seed')
    parser.add_argument('--model', default='AdaptiveFusion', type=str, help='model')
    # input
    parser.add_argument('--data_path', default='/home/nesc525/drivers/7/mmBody', type=str, help='dataset')
    parser.add_argument("--seq_idxes", type=str, default='') 
    parser.add_argument('--num_points', default=1024, type=int, help='number of points per frame')
    parser.add_argument('--normal_scale', default=1, type=int, help='normal scale of labels')
    parser.add_argument('--skip_head', default=0, type=int, help='number of skip frames')
    parser.add_argument('--skip_tail', default=0, type=int)
    parser.add_argument('--output_dim', default=148, type=int, help='output dim')
    parser.add_argument('--use_6d_pose', default=True, type=int, help='use 6d pose')
    parser.add_argument('--feature_type', default='none', type=str, help='type of features')
    parser.add_argument('--features', default=3, type=int, help='dim of features')
    parser.add_argument('--dataset', default="mmBodyDataset", type=str, help='type of input data, mmWave, Depth or mmFusion')
    parser.add_argument('--test_scene', default="lab1", type=str, help='type of test data, test, rain, smoke, night, occlusion')
    parser.add_argument('--inputs', type=str, default='image0,image1,depth0,depth1,radar', help='input data')
    parser.add_argument("--mesh_type", default='smplx', type=str, help="smplx or smpl") 
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--num_clusters', type=int, default=49)
    parser.add_argument('--mask_limbs', action="store_true")
    parser.add_argument('--point_mask_ratio', type=float, default=0.99)
    parser.add_argument('--num_mask_limbs', type=int, default=None)
    parser.add_argument('--need_augm', action="store_true")
    parser.add_argument('--mix_data', action="store_true")
    parser.add_argument('--eval_test_dataset', action="store_true")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--need_crop', action="store_true")
    # point backbone
    parser.add_argument('--radius', default=0.7, type=float, help='radius for the ball query')
    parser.add_argument('--nsample', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--npoint', default=49, type=int, help='number of points for FPS')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim_head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp_dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr_milestones', nargs='+', default=[100,200], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--use_gender', action="store_true", help='use gender')
    parser.add_argument('--fix_modalities', action="store_true")
    parser.add_argument('--device', default='cuda', type=str, help='cuda device')
    parser.add_argument('--gpu_idx', default=0, type=int, help='cuda idx')
    parser.add_argument('--joints_2d_loss', type=str, default='')
    parser.add_argument('--trans_loss', action="store_true")
    parser.add_argument('--loss_weight', default="1,1,1,1,1,1", type=str, help='weight of loss')
    parser.add_argument('--wo_MMM', action="store_true")
    # output
    parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
    parser.add_argument('--num_frames', default=10000, type=int, help='number of test frames')
    parser.add_argument('--output_dir', default='', type=str, help='path where to save')
    # others
    parser.add_argument('--resume_checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--train', dest="train", action="store_true", help='train or test')
    parser.add_argument('--visual', dest="visual", action="store_true", help='visual')
    parser.add_argument('--save_snapshot', dest="save_snapshot", action="store_true", help='save snapshot')
    parser.add_argument('--read_orig_img', action="store_true")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    main(args)
