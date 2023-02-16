import copy
import random
import torch
import numpy as np
from src.datasets.utils import gen_random_indices
from src.modeling.bert.transformer import Transformer


class AdaptiveFusion(torch.nn.Module):
    def __init__(self, args, backbone, trans_encoder):
        super(AdaptiveFusion, self).__init__()
        self.radar_backbone = backbone['radar']
        self.image_backbone = backbone['image']
        self.depth_backbone = backbone['depth']
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.radar_feat_num_dim = torch.nn.Linear(args.num_clusters, 1)
        self.image_feat_dim = torch.nn.Linear(1024, 2051)
        self.depth_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_embedding = torch.nn.Linear(3, 2051)
        self.GIM = Transformer(dim=2048, depth=3, heads=8, dim_head=256, mlp_dim=4096)
        self.global_embedding = torch.nn.Embedding(5, 2048)
        self.local_embedding = torch.nn.Embedding(5, 2051)
        self.calib_embedding = torch.nn.Linear(12, 2051)
        
        self.input_indices = []
        
    def process_image(self, images, pos_emb):
        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.image_backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.image_feat_dim(grid_feat)
        # add embeddings
        embeddings = torch.full((images.shape[0], 1), pos_emb, dtype=torch.long, device=images.device)
        global_pos_emb = self.global_embedding(embeddings)
        local_pos_emb = self.local_embedding(embeddings)
        image_feat = image_feat.view(images.shape[0],1,2048) + global_pos_emb
        grid_feat += local_pos_emb
        
        return image_feat, grid_feat

    def process_point(self, points, pos_emb):
        # extract cluster features and global point features using a PointNet backbone
        if points.shape[1] == 1024:
            xyz, cluster_feat = self.radar_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
            point_feat = self.radar_feat_num_dim(cluster_feat).squeeze()        
        else:
            xyz, cluster_feat = self.depth_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
            point_feat = self.depth_feat_num_dim(cluster_feat).squeeze()
        cluster_feat = torch.cat([xyz, cluster_feat.transpose(1,2).contiguous()], dim=2)
        # add embeddings
        xyz_embedding = self.point_embedding(xyz)
        cluster_feat += xyz_embedding
        embeddings = torch.full((points.shape[0], 1), pos_emb, dtype=torch.long, device=points.device)
        global_pos_emb = self.global_embedding(embeddings)
        local_pos_emb = self.local_embedding(embeddings)
        point_feat = point_feat.view(points.shape[0],1,2048) + global_pos_emb
        cluster_feat += local_pos_emb
            
        return point_feat, cluster_feat, xyz
    
    def forward(self, args, data_dict, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = data_dict['joints_3d'].shape[0]
        # Generate T-pose template mesh
        template_pose = torch.zeros((1, 69))
        template_pose[:, 0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(args.device)
        template_betas = torch.zeros((1, 16)).cuda(args.device)
        template_res = smpl(template_pose, template_betas)
        template_vertices = template_res['verts']

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression
        template_3d_joints = template_res['joints'][:, :22]
        template_pelvis = template_3d_joints[:, 0, :]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        if is_train:
            inputs = args.inputs.copy()
            # increase the probability of single input
            if np.random.rand() < args.single_input_ratio:
                input_mask_indices = gen_random_indices(max_random_num=len(inputs), size=4)
            else:
                input_mask_indices = gen_random_indices(max_random_num=len(inputs))
            if args.radar_only:
                input_mask_indices = [1,2,3,4]
            if args.image_only:
                input_mask_indices = [0,2,3,4]
            if args.depth_only:
                input_mask_indices = [0,1,3,4]
            self.input_indices.append(list(np.delete(np.arange(5), input_mask_indices)))
            # input mask
            for i in input_mask_indices:
                data_dict[inputs[i]] = torch.Tensor([[]])
                inputs[i] = ''
            # conduct image mask if image and point modalities are both active
            # if {'master_image','sub_image'} & set(inputs) and {'radar','master_depth','sub_depth'} & set(inputs):
            # generate mask indices of batch
            image_indices = gen_random_indices(batch_size, size=3)
            if 'master_image' in inputs:
                master_image_mask = np.ones((batch_size, 1, 1, 1))
                master_image_mask[image_indices[:2]] = 0.0
                master_image_mask = torch.from_numpy(master_image_mask).float().to(args.device)
                master_image_mask = master_image_mask.expand(-1, 3, 224, 224)
                data_dict['master_image'] *= master_image_mask
            if 'sub_image' in inputs:
                sub_image_mask = np.ones((batch_size, 1, 1, 1))
                sub_image_mask[image_indices[1:]] = 0.0
                sub_image_mask = torch.from_numpy(sub_image_mask).float().to(args.device)
                sub_image_mask = sub_image_mask.expand(-1, 3, 224, 224)
                data_dict['sub_image'] *= sub_image_mask
                
            point_indices = gen_random_indices(batch_size, size=3)
            if 'master_depth' in inputs:
                master_point_mask = np.ones((batch_size, 1, 1))
                master_point_mask[point_indices[:2]] = 0.0
                master_point_mask = torch.from_numpy(master_point_mask).float().to(args.device)
                master_point_mask = master_point_mask.expand(-1, 4096, 6)
                data_dict['master_depth'] *= master_point_mask
            if 'sub_depthe' in inputs:
                sub_point_mask = np.ones((batch_size, 1, 1))
                sub_point_mask[point_indices[1:]] = 0.0
                sub_point_mask = torch.from_numpy(sub_point_mask).float().to(args.device)
                sub_point_mask = sub_point_mask.expand(-1, 4096, 6)
                data_dict['sub_depthe'] *= sub_point_mask
        
        global_feats = []
        local_feats = []
        cluster_centers = {}
        
        # extract global and local features
        for i, input in enumerate(['radar','master_image','master_depth','sub_image','sub_depth']):
            if data_dict[input].shape[1]:
                if i == 1 or i == 3:
                    gloabl_feat, local_feat = self.process_image(data_dict[input], pos_emb=i)
                else:
                    gloabl_feat, local_feat, xyz = self.process_point(data_dict[input], pos_emb=i)
                    cluster_centers[input] = xyz
                global_feats.append(gloabl_feat)
                local_feats.append(local_feat)
                
        if is_train and args.shuffle_inputs:
            # shuffle global and local features
            feat_zip = list(zip(global_feats, local_feats))
            random.shuffle(feat_zip)
            global_feats, local_feats = zip(*feat_zip)
            
        # concatinate global and local features
        global_feats = torch.cat(global_feats, dim=1)
        local_feats = torch.cat(local_feats, dim=1)
        if not args.wo_GIM:
            # integrate global features
            global_feats = self.GIM(global_feats)
        fusion_feat = torch.sum(global_feats, dim=1, keepdim=True).expand(-1,ref_vertices.shape[-2], -1)
        # concatinate global features and 3d mesh template
        features = torch.cat([ref_vertices, fusion_feat], dim=2)
        if not args.wo_local_feat:
            # prepare input tokens including joint/vertex queries and local features
            features = torch.cat([features, local_feats], dim=1)

        if is_train:
            # vertex/joint mask
            num_mask = 677
            special_token = torch.ones_like(features[:,:num_mask,:]).cuda()*0.01
            features[:,:num_mask,:] = features[:,:num_mask,:]*meta_masks[:,:num_mask,:] + special_token*(1-meta_masks[:,:num_mask,:])

        pred_dict = {}
        # forward pass
        if args.show_att:
            features, _, att = self.trans_encoder(features)
            pred_dict['attention'] = att[-1]
            pred_dict['cluster_centers'] = cluster_centers
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:677,:]

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)
        
        return pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full

class FusionGraphormer(torch.nn.Module):
    def __init__(self, backbone, trans_encoder):
        super(AdaptiveFusion, self).__init__()
        self.radar_backbone = backbone['radar']
        self.image_backbone = backbone['image']
        self.depth_backbone = backbone['depth']
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.radar_feat_num_dim = torch.nn.Linear(49, 1)
        self.image_feat_dim = torch.nn.Linear(1024, 2051)
        self.depth_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_embedding = torch.nn.Linear(3, 2051)
        self.GIM = Transformer(dim=2048, depth=3, heads=8, dim_head=256, mlp_dim=4096)
        self.global_embedding = torch.nn.Embedding(5, 2048)
        self.local_embedding = torch.nn.Embedding(5, 2051)
        self.calib_embedding = torch.nn.Linear(12, 2051)
        
        self.input_indices = []
        
    def process_image(self, images, pos_emb):
        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.image_backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.image_feat_dim(grid_feat)
        # add embeddings
        embeddings = torch.full((images.shape[0], 1), pos_emb, dtype=torch.long, device=images.device)
        global_pos_emb = self.global_embedding(embeddings)
        local_pos_emb = self.local_embedding(embeddings)
        image_feat = image_feat.view(images.shape[0],1,2048) + global_pos_emb
        grid_feat += local_pos_emb
        
        return image_feat, grid_feat

    def process_point(self, points, pos_emb):
        # extract cluster features and global point features using a PointNet backbone
        if points.shape[1] == 1024:
            xyz, cluster_feat = self.radar_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
            point_feat = self.radar_feat_num_dim(cluster_feat).squeeze()        
        else:
            xyz, cluster_feat = self.depth_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
            point_feat = self.depth_feat_num_dim(cluster_feat).squeeze()
        cluster_feat = torch.cat([xyz, cluster_feat.transpose(1,2).contiguous()], dim=2)
        # add embeddings
        xyz_embedding = self.point_embedding(xyz)
        cluster_feat += xyz_embedding
        embeddings = torch.full((points.shape[0], 1), pos_emb, dtype=torch.long, device=points.device)
        global_pos_emb = self.global_embedding(embeddings)
        local_pos_emb = self.local_embedding(embeddings)
        point_feat = point_feat.view(points.shape[0],1,2048) + global_pos_emb
        cluster_feat += local_pos_emb
            
        return point_feat, cluster_feat, xyz
    
    def forward(self, args, data_dict, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = data_dict['joints_3d'].shape[0]
        # Generate T-pose template mesh
        template_pose = torch.zeros((1, 69))
        template_pose[:, 0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(args.device)
        template_betas = torch.zeros((1, 16)).cuda(args.device)
        template_res = smpl(template_pose, template_betas)
        template_vertices = template_res['verts']

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression
        template_3d_joints = template_res['joints'][:, :22]
        template_pelvis = template_3d_joints[:, 0, :]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        if is_train:
            inputs = args.inputs.copy()
            # increase the probability of single input
            if np.random.rand() < args.single_input_ratio:
                input_mask_indices = gen_random_indices(max_random_num=len(inputs), size=4)
            else:
                input_mask_indices = gen_random_indices(max_random_num=len(inputs))
            if args.radar_only:
                input_mask_indices = [1,2,3,4]
            if args.image_only:
                input_mask_indices = [0,2,3,4]
            if args.depth_only:
                input_mask_indices = [0,1,3,4]
            self.input_indices.append(list(np.delete(np.arange(5), input_mask_indices)))
            # input mask
            for i in input_mask_indices:
                data_dict[inputs[i]] = torch.Tensor([[]])
                inputs[i] = ''
            # conduct image mask if image and point modalities are both active
            if {'master_image','sub_image'} & set(inputs) and {'radar','master_depth','sub_depth'} & set(inputs):
                image_mask = np.ones((batch_size, 1, 1, 1))
                # generate mask indices of batch
                image_indices = gen_random_indices(batch_size, random_ratio=args.mask_ratio)
                image_mask[image_indices] = 0.0
                image_mask = torch.from_numpy(image_mask).float().to(args.device)
                image_mask = image_mask.expand(-1, 3, 224, 224)
                if 'master_image' in inputs:
                    data_dict['master_image'] *= image_mask
                if 'sub_image' in inputs:
                    data_dict['sub_image'] *= image_mask
                    
                point_mask = np.ones((batch_size, 1, 1))
                point_indices = gen_random_indices(batch_size, random_ratio=args.mask_ratio)
                point_mask[point_indices] = 0.0
                point_mask = torch.from_numpy(point_mask).float().to(args.device)
                point_mask = point_mask.expand(-1, 4096, 6)
                if 'master_depth' in inputs:
                    data_dict['master_depth'] *= point_mask
                if 'sub_depthe' in inputs:
                    data_dict['sub_depthe'] *= point_mask
        
        global_feats = []
        local_feats = []
        cluster_centers = {}
        
        # extract global and local features
        for i, input in enumerate(['radar','master_image','master_depth','sub_image','sub_depth']):
            if data_dict[input].shape[1]:
                if i == 1 or i == 3:
                    gloabl_feat, local_feat = self.process_image(data_dict[input], pos_emb=i)
                else:
                    gloabl_feat, local_feat, xyz = self.process_point(data_dict[input], pos_emb=i)
                    cluster_centers[input] = xyz
                global_feats.append(gloabl_feat)
                local_feats.append(local_feat)
                
        if is_train and args.shuffle_inputs:
            # shuffle global and local features
            feat_zip = list(zip(global_feats, local_feats))
            random.shuffle(feat_zip)
            global_feats, local_feats = zip(*feat_zip)
            
        # concatinate global and local features
        global_feats = torch.cat(global_feats, dim=1)
        local_feats = torch.cat(local_feats, dim=1)
        if not args.wo_GIM:
            # integrate global features
            global_feats = self.GIM(global_feats)
        fusion_feat = torch.sum(global_feats, dim=1, keepdim=True).expand(-1,ref_vertices.shape[-2], -1)
        # concatinate global features and 3d mesh template
        features = torch.cat([ref_vertices, fusion_feat], dim=2)
        if not args.wo_local_feat:
            # prepare input tokens including joint/vertex queries and local features
            features = torch.cat([features, local_feats], dim=1)

        if is_train:
            # vertex/joint mask
            num_mask = 677
            special_token = torch.ones_like(features[:,:num_mask,:]).cuda()*0.01
            features[:,:num_mask,:] = features[:,:num_mask,:]*meta_masks[:,:num_mask,:] + special_token*(1-meta_masks[:,:num_mask,:])

        pred_dict = {}
        # forward pass
        if args.show_att:
            features, _, att = self.trans_encoder(features)
            pred_dict['attention'] = att[-1]
            pred_dict['cluster_centers'] = cluster_centers
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:677,:]

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)
        
        return pred_dict, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full