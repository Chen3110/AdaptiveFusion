import cv2
import os.path as op
import numpy as np

import torch
import torchvision.transforms as transforms
from src.datasets.utils import *
from src.modeling._smpl import SMPL


class mmBodyDataset():
    def __init__(self, args):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.seq_idxes = eval(args.seq_idxes) if args.seq_idxes else range(20)
        self.inputs = args.inputs
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

        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = mmBodySequenceLoader(path, self.args.skip_head, self.args.skip_tail, resource=self.inputs)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
    def global_to_seq_index(self, global_idx:int):
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx
        raise IndexError
    
    def process_image(self, image, joints=None, trans_mat=None, need_crop=False, intrinsic=None):
        if need_crop:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop person area using joints
            image = crop_image(joints, image, trans_mat, square=True, intrinsic=intrinsic, visual=False)
            image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype('float32'), (2,0,1))/255.0
        image = torch.from_numpy(image).float()
        transformed_img = self.normalize_img(image)
        return transformed_img
    
    def process_pcl(self, pcl, joints, padding_points=1024, mask=False, need_filter=False, pelvis_idx=0):
        # filter person pcl using joints
        if need_filter:
            pcl = filter_pcl(joints, pcl)
        if not pcl.shape[0]:
            return torch.zeros((padding_points, 6)).float()
        if mask:
            # mask points
            limb_joints = [4, 5, 7, 8, 10, 11, 15, 18, 19, 20, 21]
            mask_joints = gen_random_indices(max_size=len(limb_joints), min_size=2)
            dists = np.minimum.reduce([np.linalg.norm(pcl[:,:3] - joints[limb_joints[i]], axis=1) for i in mask_joints])
            sele_indices = dists > 0.2
            pcl = pcl[sele_indices]
        # normalize pcl
        pcl[:,:3] -= joints[pelvis_idx]
        # padding pcl
        pcl = pad_pcl(pcl, padding_points)
        pcl = torch.from_numpy(pcl).float()
        return pcl
    
    def gen_mask(self, num_mask, mask_ratio):
        mask = np.ones((num_mask))
        if self.args.train:
            indices = gen_random_indices(num_mask, max_ratio=mask_ratio)
            mask[indices] = 0.0
        return torch.from_numpy(mask).float()
    
    def load_data(self, seq_loader, frame_idx):
        frame = seq_loader[frame_idx]
        # get mesh parameters
        mesh = dict(frame['mesh'])
        joints = mesh['joints'][:22]
        verts = mesh['vertices']
        result = {}
        
        # masking tokens
        if self.args.mesh_type == 'smplx':
            joint_mask = self.gen_mask(num_mask=22, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            vert_mask = self.gen_mask(num_mask=655, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            pelvis_idx = 0
        elif self.args.mesh_type == 'smpl':
            joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            pelvis_idx = 14
            
        # process image
        orig_imgs = {}
        trans_mat = {}
        bbox = {}
        for image in self.args.input_dict['image']:
            if self.args.show_att:
                orig_imgs[image] = frame[image]
            result[image] = self.process_image(frame[image], joints, seq_loader.calib[image])
            trans_mat[image] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[image])).float()
            bbox[image] = frame['bbox{}'.format(image[-1])].reshape(-1)

        # process depth pcl
        for depth in self.args.input_dict['depth']:
            result[depth] = self.process_pcl(frame[depth], joints, self.args.num_points, mask=self.args.mask_limbs, pelvis_idx=pelvis_idx)
            trans_mat[depth] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[depth])).float()
            
        # process radar pcl
        radar_pcl = torch.Tensor([])
        if 'radar0' in self.args.inputs:
            radar_pcl = frame['radar0']
            radar_pcl[:,3:] /= np.array([5e-38, 5., 150.])
            result['radar0'] = self.process_pcl(radar_pcl, joints, 1024, need_filter=True, pelvis_idx=pelvis_idx)
            trans_mat['radar0'] = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).float()
                            
        # return result
        result['joints_3d'] = torch.from_numpy(joints).float()
        result['root_pelvis'] = torch.from_numpy(joints[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        result['trans_mat'] = trans_mat
        result['orig_img'] = orig_imgs
        result['bbox'] = bbox
        
        return result

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader =self.seq_loaders[seq_path]
        
        return self.load_data(seq_loader, frame_idx)
    

class HuMManDataset(mmBodyDataset):
    def __init__(self, args, **kwargs):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.resource = args.inputs
        self.smpl_model = kwargs.get('smpl_model', SMPL())
        self.init_index_map()
        
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path)]
        
        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = HuMManSequenceLoader(path, resource=self.resource)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
    
    def load_data(self, seq_loader, frame_idx):
        frame = seq_loader[frame_idx]
        # get mesh parameters
        smpl_para = dict(frame['mesh'])
        pose = np.concatenate((smpl_para['transl'], smpl_para['global_orient'], smpl_para['body_pose']))
        betas = smpl_para['betas']
        mesh = self.smpl_model(torch.from_numpy(pose[None]).float(), torch.from_numpy(betas[None]).float())
        joints = copy2cpu(mesh['joints'][0])
        verts = copy2cpu(mesh['vertices'][0])
        result = {}
        
        # masking tokens
        if self.args.mesh_type == 'smplx':
            joint_mask = self.gen_mask(num_mask=22, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            vert_mask = self.gen_mask(num_mask=655, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            pelvis_idx = 0
        elif self.args.mesh_type == 'smpl':
            joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            pelvis_idx = 14
            
        trans_mat = {}
        # process image
        for image in self.args.input_dict['image']:
            result[image] = self.process_image(frame[image], joints, seq_loader.calib[image])
            trans_mat[image] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[image])).float()

        # process depth pcl
        for depth in self.args.input_dict['depth']:
            result[depth] = self.process_pcl(frame[depth], joints, self.args.num_points, mask=False, pelvis_idx=pelvis_idx)
            trans_mat[depth] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[depth])).float()

        # return result
        result['joints_3d'] = torch.from_numpy(joints).float()
        result['root_pelvis'] = torch.from_numpy(joints[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()
        result['trans_mat'] = trans_mat

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask
        
        return result


class BEHAVEDataset(mmBodyDataset):
    def __init__(self, args, **kwargs):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = op.join(args.data_path, 'train') if args.train else op.join(args.data_path, 'test')
        self.resource = args.inputs
        self.init_index_map()
        
    def init_index_map(self):
        self.index_map = [0,]
        self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path)]
        self.seq_paths.sort()
        
        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = BEHAVESequenceLoader(path, resource=self.resource)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
    
    def load_data(self, seq_loader, frame_idx):
        frame = seq_loader[frame_idx]
        # get mesh parameters
        mesh = dict(frame['mesh'])
        joints = mesh['joints']
        verts = mesh['vertices']
        result = {}
        
        # masking tokens
        if self.args.mesh_type == 'smplx':
            joint_mask = self.gen_mask(num_mask=22, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            vert_mask = self.gen_mask(num_mask=655, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            pelvis_idx = 0
        elif self.args.mesh_type == 'smpl':
            joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
            pelvis_idx = 14
            
        trans_mat = {}
        # process image
        for image in self.args.input_dict['image']:
            result[image] = self.process_image(frame[image], joints, seq_loader.calib[image])
            trans_mat[image] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[image])).float()

        # process depth pcl
        for depth in self.args.input_dict['depth']:
            result[depth] = self.process_pcl(frame[depth], joints, self.args.num_points, mask=False, pelvis_idx=pelvis_idx)
            trans_mat[depth] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[depth])).float()

        # return result
        result['joints_3d'] = torch.from_numpy(joints).float()
        result['root_pelvis'] = torch.from_numpy(joints[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()
        result['trans_mat'] = trans_mat

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        return result
    

class Human36MDataset(BEHAVEDataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        
    def init_index_map(self):
        self.index_map = [0,]
        if self.args.mix_data:
            self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path)]
        else:
            self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path) if p[0]=='s']
        self.seq_paths.sort()
        if self.args.mix_data:
            self.seq_paths = self.seq_paths[::2]
        
        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = Human36MSequenceLoader(path, resource=self.resource)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
    