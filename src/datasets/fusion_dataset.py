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
    
    def process_image(self, image, joints_3d=None, trans_mat=None, need_crop=False, intrinsic=None):
        if need_crop:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop person area using joints
            image = crop_image(joints_3d, image, trans_mat, square=True, intrinsic=intrinsic, visual=False)
            image = cv2.resize(image, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image.astype('float32'), (2,0,1))/255.0
        image = torch.from_numpy(image).float()
        transformed_img = self.normalize_img(image)
        return transformed_img
    
    def process_pcl(self, pcl, joints, padding_points=1024, mask=False, need_filter=False, pelvis_idx=0, mask_ratio=0.99, num_mask_limbs=None):
        # filter person pcl using joints
        if need_filter:
            pcl = filter_pcl(joints, pcl)
        if mask:
            # mask points
            lowwer_body_mask = pcl[:, 2] < joints[1,2] - 0.2
            limb_joints = [15, 18, 19, 20, 21]
            if num_mask_limbs is None:
                mask_ind = gen_random_indices(num_indices=len(limb_joints), min_size=0, includ_right=True)
            else:
                mask_ind = gen_random_indices(num_indices=len(limb_joints), size=num_mask_limbs)
            if len(mask_ind):
                limb_ind = np.array(limb_joints)[mask_ind]
                dists = np.minimum.reduce([np.linalg.norm(pcl[:,:3] - joints[i], axis=1) for i in limb_ind])
                points_mask_ind = np.logical_or(dists < 0.2, lowwer_body_mask)
            else:
                points_mask_ind = lowwer_body_mask
            mask_points = pcl[points_mask_ind]
            sel_res_ind = gen_random_indices(len(mask_points), max_ratio=1-mask_ratio)
            pcl = np.vstack((pcl[~points_mask_ind], mask_points[sel_res_ind]))
            
        if not pcl.shape[0]:
            return torch.zeros((padding_points, 6)).float()
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
        joints_3d = mesh['joints'][:22]
        verts = mesh['vertices']
        result = {}
        
        # masking tokens
        joint_mask = self.gen_mask(num_mask=22, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=655, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        pelvis_idx = 0
            
        # process image
        orig_imgs = {}
        trans_mat = {}
        bbox = {}
        joints_2d = {}
        for image in self.args.input_dict['image']:
            if self.args.show_att:
                orig_imgs[image] = frame[image].copy()
            result[image] = self.process_image(frame[image], joints_3d, seq_loader.calib[image])
            trans_mat[image] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[image])).float()
            bbox[image] = frame['bbox{}'.format(image[-1])].reshape(-1)
            resize_ratio = [self.img_res, self.img_res] / (bbox[image][2:] - bbox[image][:2])
            joints_2d[image] = ((project_pcl(joints_3d, trans_mat=seq_loader.calib[image], 
                                intrinsic=INTRINSIC[image]) - bbox[image][0:2]) * resize_ratio).astype(np.int64)

        # process depth pcl
        for depth in self.args.input_dict['depth']:
            result[depth] = self.process_pcl(frame[depth], joints_3d, self.args.num_points, mask=self.args.mask_limbs, 
                                             pelvis_idx=pelvis_idx, mask_ratio=self.args.point_mask_ratio,
                                             num_mask_limbs=self.args.num_mask_limbs)
            trans_mat[depth] = torch.tensor(trans_mat_2_tensor(seq_loader.calib[depth])).float()
            
        # process radar pcl
        radar_pcl = torch.Tensor([])
        if 'radar0' in self.args.inputs:
            radar_pcl = frame['radar0']
            radar_pcl[:,3:] /= np.array([5e-38, 5., 150.])
            result['radar0'] = self.process_pcl(radar_pcl, joints_3d, 1024, need_filter=True, pelvis_idx=pelvis_idx)
            trans_mat['radar0'] = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).float()
                            
        # return result
        result['joints_3d'] = torch.from_numpy(joints_3d).float()
        result['root_pelvis'] = torch.from_numpy(joints_3d[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask

        result['trans_mat'] = trans_mat
        result['orig_img'] = orig_imgs
        result['bbox'] = bbox
        result['joints_2d'] = joints_2d
        
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
        self.seq_paths = [op.join(self.data_path, p) for p in os.listdir(self.data_path) if p[0]=='s']
        self.seq_paths.sort()
        
        self.seq_loaders = {}
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = Human36MSequenceLoader(path, resource=self.resource)
            self.seq_loaders.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))


class CLIFFDataset(mmBodyDataset):
    def __init__(self, args, data_path):
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.img_res = 224
        self.data_path = op.join(data_path, 'train')
        self.images = os.listdir(os.path.join(self.data_path, 'crop_images'))
        self.labels = os.listdir(os.path.join(self.data_path, 'mesh'))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        result = {}
        mesh = dict(np.load(os.path.join(self.data_path, 'mesh', self.labels[idx])))
        joints = mesh['joints']
        verts = mesh['vertices']
        
        # masking tokens
        joint_mask = self.gen_mask(num_mask=24, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        vert_mask = self.gen_mask(num_mask=431, mask_ratio=self.args.mask_ratio).unsqueeze(-1)
        pelvis_idx = 14
        
        # process image
        img = cv2.imread(os.path.join(self.data_path, 'crop_images', self.images[idx]))
        result[self.args.inputs[0]] = self.process_image(img)
        trans_mat = {self.args.inputs[0]:torch.ones([4,4])}
        
        # return result
        result['joints_3d'] = torch.from_numpy(joints).float()
        result['root_pelvis'] = torch.from_numpy(joints[pelvis_idx]).float()
        result['vertices'] = torch.from_numpy(verts).float()
        result['trans_mat'] = trans_mat

        result['joint_mask'] = joint_mask
        result['vert_mask'] = vert_mask
        return result
    