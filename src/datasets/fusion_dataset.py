"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import cv2
import os.path as op
import numpy as np

import torch
import torchvision.transforms as transforms
from src.datasets.utils import *


class FusionDataset():
    def __init__(self, args, **kwargs):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.seq_idxes = eval(args.seq_idxes) if args.seq_idxes else range(20)
        self.args = args
        self.init_index_map()
            
    def init_index_map(self):
        self.index_map = [0,]
        if self.args.train:
            seq_dirs = ['sequence_{}'.format(i) for i in self.seq_idxes]
            self.seq_paths = [op.join(self.data_path, p) for p in seq_dirs]
        else:
            seq_dirs = ['sequence_{}'.format(i) for i in range(2)]
            self.seq_paths = [op.join(self.data_path, self.args.test_scene, p) for p in seq_dirs]
        
        print('Data path: ', self.seq_paths)

        self.seq_loaders = []
        self.full_data = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = Sequence_Loader(path, self.args.skip_head, self.args.skip_tail, resource=self.args.resource)
            self.full_data.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
    def global_to_seq_index(self, global_idx:int):
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx
        raise IndexError
        
    def mosh_pose_transform(self, trans, root_orient, root_joint, trans_mat):
        mosh_offset = root_joint - trans
        new_trans = trans_mat['R'] @ (trans + mosh_offset) + trans_mat['t'] - mosh_offset
        orient_mat = trans_mat['R'] @ cv2.Rodrigues(root_orient)[0]
        new_orient = cv2.Rodrigues(orient_mat)[0]
        return new_trans, new_orient
    
    def process_image(self, image, joints=None, need_crop=False):
        if need_crop:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop person area using joints
            image = crop_image(joints, image, square=True)
            image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype('float32'), (2,0,1))/255.0
        image = torch.from_numpy(image).float()
        transformed_img = self.normalize_img(image)
        return transformed_img
    
    def process_pcl(self, pcl, joints, trans_mat, padding_points=1024, mask_ratio=0.):
        # filter person pcl using joints
        pcl = filter_pcl(joints, pcl)
        if not pcl.shape[0]:
            pcl = np.zeros([1, 6], dtype=np.float32)
        if mask_ratio:
            # mask x% of points
            mask_indices = gen_mask_indices(pcl.shape[0], mask_ratio, random_num=False)
            pcl = np.delete(pcl, mask_indices, axis=0)
        if trans_mat is not None:
            trans_pcl = (pcl[:,:3] - trans_mat['t']) @ trans_mat['R']
            pcl = np.hstack((trans_pcl, pcl[:,3:]))
        # padding pcl
        pcl = pad_pcl(pcl, padding_points)
        pcl = torch.from_numpy(pcl).float()
        return pcl
    
    def gen_mask(self, num_mask, mask_ratio):
        mask = np.ones((num_mask))
        if self.args.train:
            indices = gen_mask_indices(num_mask, mask_ratio)
            mask[indices] = 0.0
        return torch.from_numpy(mask).float()
    
    def load_data(self, seq_loader, frame_idx):
        if isinstance(seq_loader, list):
            return seq_loader[frame_idx]
        frame = seq_loader[frame_idx]
        # get mesh parameters
        mesh = dict(frame['mesh'])
        pose = mesh['pose']
        betas = mesh['shape']
        gender = 'None'
        joints = mesh['joints'][:22]

        # process image
        mas_img = np.array([])
        if 'master_image' in self.args.resource:
            trans_mat_mas = seq_loader.calib['kinect_master']
            joints_mas = (joints - trans_mat_mas['t']) @ trans_mat_mas['R']
            mas_img = self.process_image(frame['master_image'], joints_mas)

        sub_img = np.array([])
        if 'sub_image' in self.args.resource:
            trans_mat_sub = seq_loader.calib['kinect_sub']
            joints_sub = (joints - trans_mat_sub['t']) @ trans_mat_sub['R']
            sub_img = self.process_image(frame['sub_image'], joints_sub)
        
        # transform mesh param and joints to the kinect coordinate if need
        trans_joints = joints
        trans_pose = pose
        trans_mat_to_cam = None
        if self.args.trans_coor_to_cam:
            if 'master_image' in self.args.resource:
                trans_mat_to_cam = trans_mat_mas
                trans_joints = joints_mas
            elif 'sub_image' in self.args.resource:
                trans_mat_to_cam = trans_mat_sub
                trans_joints = joints_sub
            else:
                raise RuntimeError("Kinect image must in resource.")
            rev_trans_mat = {'R':np.asarray(trans_mat_to_cam['R']).T, 't':-np.asarray(trans_mat_to_cam['R']).T @ trans_mat_to_cam['t']}
            trans, root_orient = self.mosh_pose_transform(pose[:3], pose[3:6], joints[0], rev_trans_mat)
            trans_pose = np.hstack((trans, root_orient.reshape(-1), pose[6:]))
                
        # get 3d joints center
        joint_center = ((trans_joints.max(axis=0) + trans_joints.min(axis=0))/2)[:3]
        
        # process radar pcl
        radar_pcl = np.array([])
        if 'radar' in self.args.resource:
            radar_pcl = frame['radar']
            radar_pcl[:,3:] /= np.array([5e-38, 5., 150.])
            radar_pcl = self.process_pcl(radar_pcl, joints, trans_mat_to_cam)
            
        # process depth pcl
        mask_ratio = 0.99 if self.args.train else 0.
        mas_depth = np.array([])
        if 'master_depth' in self.args.resource:
            mas_depth = frame['master_depth']
            mas_depth = self.process_pcl(mas_depth, joints, trans_mat_to_cam, padding_points=4096, mask_ratio=mask_ratio)
        
        sub_depth = np.array([])
        if 'sub_depth' in self.args.resource:
            sub_depth = frame['sub_depth']
            sub_depth = self.process_pcl(sub_depth, joints, trans_mat_to_cam, padding_points=4096, mask_ratio=mask_ratio)
            
        # masking tokens
        joint_mask = self.gen_mask(num_mask=22, mask_ratio=0.3).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=655, mask_ratio=0.3).unsqueeze(-1)
        
        # return result
        result = {}
        result['pose'] = torch.from_numpy(trans_pose).float()
        result['betas'] = torch.from_numpy(betas).float()
        result['joints_3d'] = torch.from_numpy(trans_joints).float()
        result['root_pelvis'] = torch.from_numpy(trans_joints[0]).float()

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        result['center'] = np.asarray(joint_center).astype(np.float32)
        result['gender'] = gender
        result['radar'] = radar_pcl
        result['master_image'] = mas_img
        result['sub_image'] = sub_img
        result['master_depth'] = mas_depth
        result['sub_depth'] = sub_depth
        
        return result

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader =self.full_data[seq_path]
        
        return self.load_data(seq_loader, frame_idx)