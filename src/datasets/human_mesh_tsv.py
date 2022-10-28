"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

from multiprocessing import Process
import pickle
import cv2
import math
import json
from PIL import Image
import os.path as op
import numpy as np
import code

from src.utils.tsv_file import TSVFile, CompositeTSVFile
from src.utils.tsv_file_ops import load_linelist_file, load_from_yaml_file, find_file_path_in_yaml
from src.utils.image_ops import img_from_base64, crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import torch
import torchvision.transforms as transforms
from src.datasets.utils import *

class MeshTSVDataset(object):
    def __init__(self, img_file, label_file=None, hw_file=None,
                 linelist_file=None, is_train=True, cv2_output=False, scale_factor=1):

        self.img_file = img_file
        self.label_file = label_file
        self.hw_file = hw_file
        self.linelist_file = linelist_file
        self.img_tsv = self.get_tsv_file(img_file)
        self.label_tsv = None if label_file is None else self.get_tsv_file(label_file)
        self.hw_tsv = None if hw_file is None else self.get_tsv_file(hw_file)

        if self.is_composite:
            assert op.isfile(self.linelist_file)
            self.line_list = [i for i in range(self.hw_tsv.num_rows())]
        else:
            self.line_list = load_linelist_file(linelist_file)

        self.cv2_output = cv2_output
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.is_train = is_train
        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 30 # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = 224

        self.image_keys = self.prepare_image_keys()

        self.joints_definition = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
        self.pelvis_index = self.joints_definition.index('Pelvis')

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file, self.linelist_file,
                        root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def get_valid_tsv(self):
        # sorted by file size
        if self.hw_tsv:
            return self.hw_tsv
        if self.label_tsv:
            return self.label_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
	    
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
	    
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
	
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [self.img_res, self.img_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_line_no(self, idx):
        return idx if self.line_list is None else self.line_list[idx]

    def get_image(self, idx): 
        line_no = self.get_line_no(idx)
        row = self.img_tsv[line_no]
        # use -1 to support old format with multiple columns.
        cv2_im = img_from_base64(row[-1])
        if self.cv2_output:
            return cv2_im.astype(np.float32, copy=True)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

        return cv2_im

    def get_annotations(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv[line_no]
            annotations = json.loads(row[1])
            return annotations
        else:
            return []

    def get_target_from_annotations(self, annotations, img_size, idx):
        # This function will be overwritten by each dataset to 
        # decode the labels to specific formats for each task. 
        return annotations


    def get_img_info(self, idx):
        if self.hw_tsv is not None:
            line_no = self.get_line_no(idx)
            row = self.hw_tsv[line_no]
            try:
                # json string format with "height" and "width" being the keys
                return json.loads(row[1])[0]
            except ValueError:
                # list of strings representing height and width in order
                hw_str = row[1].split(' ')
                hw_dict = {"height": int(hw_str[0]), "width": int(hw_str[1])}
                return hw_dict

    def get_img_key(self, idx):
        line_no = self.get_line_no(idx)
        # based on the overhead of reading each row.
        if self.hw_tsv:
            return self.hw_tsv[line_no][0]
        elif self.label_tsv:
            return self.label_tsv[line_no][0]
        else:
            return self.img_tsv[line_no][0]

    def __len__(self):
        if self.line_list is None:
            return self.img_tsv.num_rows() 
        else:
            return len(self.line_list)

    def __getitem__(self, idx):

        img = self.get_image(idx)
        img_key = self.get_img_key(idx)
        annotations = self.get_annotations(idx)

        annotations = annotations[0]
        center = annotations['center']
        scale = annotations['scale']
        has_2d_joints = annotations['has_2d_joints']
        has_3d_joints = annotations['has_3d_joints']
        joints_2d = np.asarray(annotations['2d_joints'])
        joints_3d = np.asarray(annotations['3d_joints'])

        if joints_2d.ndim==3:
            joints_2d = joints_2d[0]
        if joints_3d.ndim==3:
            joints_3d = joints_3d[0]

        # Get SMPL parameters, if available
        has_smpl = np.asarray(annotations['has_smpl'])
        pose = np.asarray(annotations['pose'])
        betas = np.asarray(annotations['betas'])

        try:
            gender = annotations['gender']
        except KeyError:
            gender = 'none'

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        transfromed_img = self.normalize_img(img)

        # normalize 3d pose by aligning the pelvis as the root (at origin)
        root_pelvis = joints_3d[self.pelvis_index,:-1]
        joints_3d[:,:-1] = joints_3d[:,:-1] - root_pelvis[None,:]
        # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
        joints_3d_transformed = self.j3d_processing(joints_3d.copy(), rot, flip)
        # 2d pose augmentation
        joints_2d_transformed = self.j2d_processing(joints_2d.copy(), center, sc*scale, rot, flip)

        ###################################
        # Masking percantage
        # We observe that 30% works better for human body mesh. Further details are reported in the paper.
        mvm_percent = 0.3
        ###################################
        
        mjm_mask = np.ones((14,1))
        if self.is_train:
            num_joints = 14
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
            mjm_mask[indices,:] = 0.0
        mjm_mask = torch.from_numpy(mjm_mask).float()

        mvm_mask = np.ones((431,1))
        if self.is_train:
            num_vertices = 431
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()

        meta_data = {}
        meta_data['ori_img'] = img
        meta_data['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        meta_data['betas'] = torch.from_numpy(betas).float()
        meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
        meta_data['has_3d_joints'] = has_3d_joints
        meta_data['has_smpl'] = has_smpl

        meta_data['mjm_mask'] = mjm_mask
        meta_data['mvm_mask'] = mvm_mask

        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = has_2d_joints
        meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
        meta_data['scale'] = float(sc * scale)
        meta_data['center'] = np.asarray(center).astype(np.float32)
        meta_data['gender'] = gender
        return img_key, transfromed_img, meta_data



class MeshTSVYamlDataset(MeshTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, yaml_file, is_train=True, cv2_output=False, scale_factor=1):
        self.cfg = load_from_yaml_file(yaml_file)
        self.is_composite = self.cfg.get('composite', False)
        self.root = op.dirname(yaml_file)
        
        if self.is_composite==False:
            img_file = find_file_path_in_yaml(self.cfg['img'], self.root)
            label_file = find_file_path_in_yaml(self.cfg.get('label', None),
                                                self.root)
            hw_file = find_file_path_in_yaml(self.cfg.get('hw', None), self.root)
            linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                                self.root)
        else:
            img_file = self.cfg['img']
            hw_file = self.cfg['hw']
            label_file = self.cfg.get('label', None)
            linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                                self.root)

        super(MeshTSVYamlDataset, self).__init__(
            img_file, label_file, hw_file, linelist_file, is_train, cv2_output=cv2_output, scale_factor=scale_factor)


class ImmDataset(MeshTSVDataset):
    def __init__(self, data_path, is_train=True, test_scene='lab1', cv2_output=False, **kwargs):
        self.cv2_output = cv2_output
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.is_train = is_train
        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 30 # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = 224
        
        self.data_path = data_path
        self.data_path += '/train' if self.is_train else '/test'
        self.test_scene = test_scene
        self.num_points = kwargs.get('num_points', 1024)
        self.features = kwargs.get('features', 3)
        self.pelvis_index = 0
        self.skip_head = kwargs.get('skip_head', 0)
        self.skip_tail = kwargs.get('skip_tail', 0)
        self.use_pkl = kwargs.get('use_pkl', True)
        self.create_pkl = kwargs.get('create_pkl', True)
        self.seq_idxes = kwargs.get('seq_idxes', range(20))
        self.add_rgb = kwargs.get('add_rgb', False)
        self.init_index_map()

    def init_index_map(self):
        self.index_map = [0,]
        if self.is_train:
            seq_dirs = ['sequence_{}'.format(i) for i in self.seq_idxes]
            self.seq_paths = [os.path.join(self.data_path, p) for p in seq_dirs]
        else:
            seq_dirs = ['sequence_{}'.format(i) for i in range(2)]
            # seq_dirs = ['sequence_{}'.format(i) for i in range(1,2)]
            self.seq_paths = [os.path.join(self.data_path, self.test_scene, p) for p in seq_dirs]
        
        print('Data path: ', self.seq_paths)

        self.seq_loaders = []
        process_dict = {}
        self.full_data = {}
        for path in self.seq_paths:
            seq_pkl = os.path.join(path, 'pkl_data/gphmr.pkl')
            try:
                if self.use_pkl and os.path.exists(seq_pkl):
                    with open(seq_pkl, 'rb') as f:
                        seq_loader = pickle.load(f)[self.skip_head:-self.skip_tail-1]
                else:
                    # init result loader, reindex
                    seq_loader = Sequence_Loader(path, self.skip_head, self.skip_tail)
                    if self.create_pkl:
                        p = Process(target=self.write_pkl, args=(path, seq_pkl, seq_loader))
                        p.start()
                        process_dict.update({path: p})
            except Exception as e:
                print(e)
                continue

            self.full_data.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
        if self.create_pkl:
            for path, p in process_dict.items():
                p.join()
                with open(os.path.join(path, 'pkl_data/gphmr.pkl'), 'rb') as f:
                    self.full_data[path] = pickle.load(f)

    def write_pkl(self, path, pkl_fname, seq_loader):
        import time
        data = []
        print('Creating pkl data of {}'.format(path))
        t_s = time.time()
        for i in range(len(seq_loader)):
            data.append(self.load_data(seq_loader, i))
            print(path, i, t_e-t_s, 's')
        if not os.path.exists(os.path.join(path, 'pkl_data')):
            os.mkdir(os.path.join(path, 'pkl_data'))
        with open(pkl_fname, 'wb') as f:
            pickle.dump(data, f)
        t_e = time.time()
        print(path, 'Done, time:{}'.format(t_e-t_s))

    def global_to_seq_index(self, global_idx:int):
        for seq_idx in range(len(self.index_map)-1):
            if global_idx in range(self.index_map[seq_idx], self.index_map[seq_idx+1]):
                frame_idx = global_idx - self.index_map[seq_idx]
                return seq_idx, frame_idx
        raise IndexError

    def pad_pcl(self, data, return_choices=False):
        if data.shape[0] > self.num_points:
            r = np.random.choice(data.shape[0], size=self.num_points, replace=False)
        else:
            repeat, residue = self.num_points // data.shape[0], self.num_points % data.shape[0]
            r = np.random.choice(data.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(data.shape[0]) for _ in range(repeat)] + [r], axis=0)
        if return_choices:
            return data[r, :], r
        return data[r, :]
        
    def mosh_pose_transform(self, trans, root_orient, root_joint, trans_mat):
        mosh_offset = root_joint - trans
        new_trans = trans_mat['R'] @ (trans + mosh_offset) + trans_mat['t'] - mosh_offset
        orient_mat = trans_mat['R'] @ cv2.Rodrigues(root_orient)[0]
        new_orient = cv2.Rodrigues(orient_mat)[0]
        return new_trans, new_orient
    
    def load_data(self, seq_loader, frame_idx):
        if isinstance(seq_loader, list):
            return seq_loader[frame_idx]
        frame = seq_loader[frame_idx]
        image = frame['master_image']
        orig_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        radar_pcl = frame['radar']
        mesh = dict(frame['mesh'])
        joints = mesh['joints'][:22]
        trans_mat = seq_loader.calib['kinect_master']
        trans_joints = (joints - trans_mat['t']) @ trans_mat['R']

        center = ((trans_joints.max(axis=0) + trans_joints.min(axis=0))/2)[:3]
        scale = 1
        has_2d_joints = True
        has_3d_joints = True
        joints_2d = project_pcl(trans_joints)
        joints_3d = trans_joints

        # Get SMPL parameters, if available
        has_smpl = True
        pose = mesh['pose']
        betas = mesh['shape']
        # transform mesh param to kinect master coordinate
        rev_trans_mat = {'R':np.asarray(trans_mat['R']).T, 't':-np.asarray(trans_mat['R']).T@trans_mat['t']}
        trans, root_orient = self.mosh_pose_transform(pose[:3], pose[3:6], joints[0], rev_trans_mat)
        pose = np.hstack((trans, root_orient.reshape(-1), pose[6:]))

        gender = 'none'

        # Get augmentation parameters
        # flip,pn,rot,sc = self.augm_params()

        # Process image
        # img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = crop_image(trans_joints, orig_img, square=True)
        img = cv2.resize(img, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img.astype('float32'),(2,0,1))/255.0
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        transformed_img = self.normalize_img(img)

        # normalize 3d pose by aligning the pelvis as the root (at origin)
        # root_pelvis = joints_3d[self.pelvis_index]
        # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
        # joints_3d_transformed = self.j3d_processing(joints_3d.copy(), rot, flip)
        joints_3d_transformed = joints_3d
        # 2d pose augmentation
        # joints_2d_transformed = self.j2d_processing(joints_2d.copy(), center, sc*scale, rot, flip)
        joints_2d_transformed = joints_2d

        # filter person pcl
        filterd_pcl = filter_pcl(joints, radar_pcl)
        trans_pcl = (filterd_pcl[:,:3] - trans_mat['t']) @ trans_mat['R']
        if self.add_rgb:
            trans_pcl = get_rgb_value(trans_pcl, orig_img)
        # padding pcl
        padding_pcl = self.pad_pcl(np.hstack((trans_pcl, filterd_pcl[:,3:]/np.array([5e-38, 5., 150.]))))
        padding_pcl = torch.from_numpy(padding_pcl).float()

        ###################################
        # Masking percantage
        # We observe that 30% works better for human body mesh. Further details are reported in the paper.
        mvm_percent = 0.3
        ###################################

        num_joints = 22
        mjm_mask = np.ones((num_joints,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
            mjm_mask[indices,:] = 0.0
        mjm_mask = torch.from_numpy(mjm_mask).float()
        
        num_vertices = 655
        mvm_mask = np.ones((num_vertices,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()
        
        num_grid_feat = 49
        grid_mask = np.ones((num_grid_feat,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_grid_feat) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_grid_feat),replace=False,size=masked_num)
            grid_mask[indices,:] = 0.0
        grid_mask = torch.from_numpy(grid_mask).float()
        
        num_cluster_feat = 32
        cluster_mask = np.ones((num_cluster_feat,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * num_cluster_feat) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_cluster_feat),replace=False,size=masked_num)
            cluster_mask[indices,:] = 0.0
        cluster_mask = torch.from_numpy(cluster_mask).float()

        meta_data = {}
        meta_data['orig_img'] = image
        meta_data['pose'] = torch.from_numpy(pose).float()
        meta_data['betas'] = torch.from_numpy(betas).float()
        meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
        meta_data['root_pelvis'] = torch.from_numpy(joints_3d_transformed[0]).float()
        meta_data['has_3d_joints'] = has_3d_joints
        meta_data['has_smpl'] = has_smpl

        meta_data['mjm_mask'] = mjm_mask
        meta_data['mvm_mask'] = mvm_mask
        meta_data['grid_mask'] = grid_mask
        meta_data['cluster_mask'] = cluster_mask

        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = has_2d_joints
        meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
        meta_data['scale'] = float(scale)
        meta_data['center'] = np.asarray(center).astype(np.float32)
        meta_data['gender'] = gender

        return padding_pcl, transformed_img, meta_data

    def __len__(self):
        return self.index_map[-1]

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.global_to_seq_index(idx)
        seq_path = self.seq_paths[seq_idx]
        seq_loader =self.full_data[seq_path]
        
        return self.load_data(seq_loader, frame_idx)



class DepthBodyDataset(ImmDataset):
    def __init__(self, data_path, is_train=True, test_scene='lab1', cv2_output=False, **kwargs):
        super().__init__(data_path, is_train, test_scene, cv2_output, **kwargs)
            
    def init_index_map(self):
        self.index_map = [0,]
        if self.is_train:
            seq_dirs = ['sequence_{}'.format(i) for i in self.seq_idxes]
            self.seq_paths = [os.path.join(self.data_path, p) for p in seq_dirs]
        else:
            seq_dirs = ['sequence_{}'.format(i) for i in range(2)]
            self.seq_paths = [os.path.join(self.data_path, self.test_scene, p) for p in seq_dirs]
        
        print('Data path: ', self.seq_paths)

        self.seq_loaders = []
        self.full_data = {}
        source = ['depth_pcl', 'image', 'mesh']
        for path in self.seq_paths:
            # init result loader, reindex
            seq_loader = Sequence_Loader(path, self.skip_head, self.skip_tail, enabled_source=source)
            self.full_data.update({path:seq_loader})
            self.index_map.append(self.index_map[-1] + len(seq_loader))
            
        
    def load_data(self, seq_loader, frame_idx):
        if isinstance(seq_loader, list):
            return seq_loader[frame_idx]
        frame = seq_loader[frame_idx]
        img = cv2.cvtColor(frame['image'], cv2.COLOR_BGR2RGB)
        depth_pcl = frame['depth_pcl']
        mesh = dict(frame['mesh'])
        joints = mesh['joints'][:22]
        trans_mat = seq_loader.calib
        trans_joints = (joints - trans_mat['t']) @ trans_mat['R']

        center = ((trans_joints.max(axis=0) + trans_joints.min(axis=0))/2)[:3]
        scale = 1
        has_2d_joints = True
        has_3d_joints = True
        joints_2d = project_pcl(trans_joints)
        joints_3d = trans_joints

        # Get SMPL parameters, if available
        has_smpl = True
        pose = mesh['pose']
        betas = mesh['shape']
        # transform mesh param to kinect master coordinate
        rev_trans_mat = {'R':np.asarray(trans_mat['R']).T, 't':-np.asarray(trans_mat['R']).T@trans_mat['t']}
        trans, root_orient = self.mosh_pose_transform(pose[:3], pose[3:6], joints[0], rev_trans_mat)
        pose = np.hstack((trans, root_orient.reshape(-1), pose[6:]))

        gender = 'none'

        # Process image
        img = crop_image(trans_joints, img)
        img = cv2.resize(img, [self.img_res, self.img_res], interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img.astype('float32'),(2,0,1))/255.0
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        transformed_img = self.normalize_img(img)

        joints_3d_transformed = joints_3d
        joints_2d_transformed = joints_2d

        # filter person pcl
        filterd_pcl = filter_pcl(joints, depth_pcl)
        if not filterd_pcl.shape[0]:
            filterd_pcl = np.zeros([1, 6], dtype=np.float32)
        trans_pcl = (filterd_pcl[:,:3] - trans_mat['t']) @ trans_mat['R']
        # padding pcl
        padding_pcl = self.pad_pcl(np.hstack((trans_pcl, filterd_pcl[:,3:])))
        padding_pcl = torch.from_numpy(padding_pcl).float()

        ###################################
        # Masking percantage
        # We observe that 30% works better for human body mesh. Further details are reported in the paper.
        mvm_percent = 0.3
        ###################################

        num_joints = 22
        mjm_mask = np.ones((num_joints,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
            mjm_mask[indices,:] = 0.0
        mjm_mask = torch.from_numpy(mjm_mask).float()
        
        num_vertices = 655
        mvm_mask = np.ones((num_vertices,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()

        num_grid_feat = 49
        grid_mask = np.ones((num_grid_feat,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * 0 * num_grid_feat) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_grid_feat),replace=False,size=masked_num)
            grid_mask[indices,:] = 0.0
        grid_mask = torch.from_numpy(grid_mask).float()
        
        num_cluster_feat = 32
        cluster_mask = np.ones((num_cluster_feat,1))
        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * num_cluster_feat) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_cluster_feat),replace=False,size=masked_num)
            cluster_mask[indices,:] = 0.0
        cluster_mask = torch.from_numpy(cluster_mask).float()
        
        meta_data = {}
        # meta_data['ori_img'] = img
        meta_data['pose'] = torch.from_numpy(pose).float()
        meta_data['betas'] = torch.from_numpy(betas).float()
        meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
        meta_data['root_pelvis'] = torch.from_numpy(joints_3d_transformed[0]).float()
        meta_data['has_3d_joints'] = has_3d_joints
        meta_data['has_smpl'] = has_smpl

        meta_data['mjm_mask'] = mjm_mask
        meta_data['mvm_mask'] = mvm_mask
        meta_data['grid_mask'] = grid_mask
        meta_data['cluster_mask'] = cluster_mask

        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = has_2d_joints
        meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
        meta_data['scale'] = float(scale)
        meta_data['center'] = np.asarray(center).astype(np.float32)
        meta_data['gender'] = gender

        return padding_pcl, transformed_img, meta_data
