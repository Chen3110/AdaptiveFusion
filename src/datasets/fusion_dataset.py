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
        self.resource = args.inputs if not args.show_att else ['radar','master_image','master_depth','sub_image','sub_depth']
        self.init_index_map()
            
    def init_index_map(self):
        self.index_map = [0,]
        if self.args.train:
            seq_dirs = ['sequence_{}'.format(i) for i in self.seq_idxes]
            self.seq_paths = [op.join(self.data_path, p) for p in seq_dirs]
        else:
            seq_dirs = ['sequence_{}'.format(i) for i in range(2)]
            # seq_dirs = ['sequence_1']
            self.seq_paths = [op.join(self.data_path, self.args.test_scene, p) for p in seq_dirs]
        
        print('Data path: ', self.seq_paths)

        self.seq_loaders = []
        self.full_data = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = SequenceLoader(path, self.args.skip_head, self.args.skip_tail, resource=self.resource)
            self.full_data.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
    def global_to_seq_index(self, global_idx:int):
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx
        raise IndexError
    
    def process_image(self, image, joints=None, trans_mat=None, need_crop=False):
        if need_crop:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop person area using joints
            image = crop_image(joints, image, trans_mat, square=True)
            image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype('float32'), (2,0,1))/255.0
        image = torch.from_numpy(image).float()
        transformed_img = self.normalize_img(image)
        return transformed_img
    
    def process_pcl(self, pcl, joints, padding_points=1024, mask=False):
        # filter person pcl using joints
        pcl = filter_pcl(joints, pcl)
        if not pcl.shape[0]:
            return torch.zeros((padding_points, 6)).float()
        if mask:
            # mask points
            indices = np.random.choice(pcl.shape[0], 100)
            pcl = pcl[indices]
        # normalize pcl
        pcl[:,:3] -= joints[0]
        # padding pcl
        pcl = pad_pcl(pcl, padding_points)
        pcl = torch.from_numpy(pcl).float()
        return pcl
    
    def gen_mask(self, num_mask, mask_ratio):
        mask = np.ones((num_mask))
        if self.args.train:
            indices = gen_random_indices(num_mask, random_ratio=mask_ratio)
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
        joints = mesh['joints'][:22]
        orig_imgs = {}
        if self.args.show_att:
            orig_imgs['master_image'] = frame['master_image']
            orig_imgs['sub_image'] = frame['sub_image']
            
        # process image
        mas_img = torch.Tensor([])
        trans_mat_mas = seq_loader.calib['kinect_master']
        if 'master_image' in self.args.inputs:
            mas_img = self.process_image(frame['master_image'], joints, trans_mat_mas, self.args.show_att)

        sub_img = torch.Tensor([])
        trans_mat_sub = seq_loader.calib['kinect_sub']
        if 'sub_image' in self.args.inputs:
            sub_img = self.process_image(frame['sub_image'], joints, trans_mat_sub, self.args.show_att)
        
        # get 3d joints center
        joint_center = ((joints.max(axis=0) + joints.min(axis=0))/2)[:3]
        
        # process radar pcl
        radar_pcl = torch.Tensor([])
        if 'radar' in self.args.inputs:
            radar_pcl = frame['radar']
            radar_pcl[:,3:] /= np.array([5e-38, 5., 150.])
            radar_pcl = self.process_pcl(radar_pcl, joints, padding_points=1024)
            
        # process depth pcl
        if self.args.train:
            random_value = np.random.rand()
            mask = True if random_value < self.args.mask_ratio else False
        else:
            mask = False
        mas_depth = torch.Tensor([])
        if 'master_depth' in self.args.inputs:
            mas_depth = self.process_pcl(frame['master_depth'], joints, padding_points=4096, mask=mask)
        
        sub_depth = torch.Tensor([])
        if 'sub_depth' in self.args.inputs:
            sub_depth = self.process_pcl(frame['sub_depth'], joints, padding_points=4096, mask=mask)
            
        # masking tokens
        joint_mask = self.gen_mask(num_mask=22, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=655, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
                
        # transform matrix
        trans_mat_mas = torch.tensor(trans_mat_2_tensor(trans_mat_mas)).float()
        trans_mat_sub = torch.tensor(trans_mat_2_tensor(trans_mat_sub)).float()
        trans_mat = {
            'radar': torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).float(),
            'master_image': trans_mat_mas,
            'master_depth': trans_mat_mas,
            'sub_image': trans_mat_sub,
            'sub_depth': trans_mat_sub,
        }
        
        # return result
        result = {}
        result['pose'] = torch.from_numpy(pose).float()
        result['betas'] = torch.from_numpy(betas).float()
        result['joints_3d'] = torch.from_numpy(joints).float()
        result['root_pelvis'] = torch.from_numpy(joints[0]).float()

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        result['center'] = np.asarray(joint_center).astype(np.float32)
        result['radar'] = radar_pcl
        result['master_image'] = mas_img
        result['sub_image'] = sub_img
        result['master_depth'] = mas_depth
        result['sub_depth'] = sub_depth
        result['trans_mat'] = trans_mat
        result['orig_img'] = orig_imgs
        
        return result

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader =self.full_data[seq_path]
        
        return self.load_data(seq_loader, frame_idx)