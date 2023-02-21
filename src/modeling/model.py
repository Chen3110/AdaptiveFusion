import copy
import random
import torch
import numpy as np
from src.datasets.utils import gen_random_indices, project_pcl_torch
from src.modeling.bert.transformer import FFN, PredictorLG, Transformer, Transformer2


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
            # Modality Masking Module
            if not args.wo_MMM:
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


class DeepFusion(torch.nn.Module):
    def __init__(self, args, backbone, trans_encoder):
        super(DeepFusion, self).__init__()
        self.radar_backbone = backbone['radar']
        self.image_backbone = backbone['image']
        self.depth_backbone = backbone['depth']
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.radar_feat_num_dim = torch.nn.Linear(args.num_clusters, 1)
        self.image_feat_dim = torch.nn.Linear(1024, 2048)
        self.depth_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_embedding = torch.nn.Linear(3, 2048)
        self.transformer = Transformer(dim=2048, depth=3, heads=8, dim_head=256, mlp_dim=4096)
        self.global_embedding = torch.nn.Embedding(5, 2048)
        self.local_embedding = torch.nn.Embedding(5, 2048)
        
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
        local_pos_emb = self.local_embedding(embeddings)
        grid_feat += local_pos_emb
        
        return grid_feat

    def process_point(self, points, pos_emb):
        # extract cluster features and global point features using a PointNet backbone
        if points.shape[1] == 1024:
            xyz, cluster_feat = self.radar_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
        else:
            xyz, cluster_feat = self.depth_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
        # add embeddings
        xyz_embedding = self.point_embedding(xyz)
        cluster_feat = cluster_feat.transpose(1,2).contiguous() + xyz_embedding
        embeddings = torch.full((points.shape[0], 1), pos_emb, dtype=torch.long, device=points.device)
        local_pos_emb = self.local_embedding(embeddings)
        cluster_feat += local_pos_emb
        
        return cluster_feat
    
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
                
            # Modality Masking Module
            if not args.wo_MMM:
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
                    
        local_feats = []
        cluster_centers = {}
        
        # extract local features
        for i, input in enumerate(['radar','master_image','master_depth','sub_image','sub_depth']):
            if data_dict[input].shape[1]:
                if i == 1 or i == 3:
                    local_feat = self.process_image(data_dict[input], pos_emb=i)
                else:
                    local_feat = self.process_point(data_dict[input], pos_emb=i)
                local_feats.append(local_feat)
            
        # concatinate local features
        local_feats = torch.cat(local_feats, dim=1)
        fusion_feats = self.transformer(local_feats)
        global_feats = torch.max(input=fusion_feats, dim=1, keepdim=True, out=None)[0]
        global_feats = global_feats.expand(-1, ref_vertices.shape[-2], -1)
        # concatinate global features and 3d mesh template
        features = torch.cat([ref_vertices, global_feats], dim=2)

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


class TokenFusion(torch.nn.Module):
    def __init__(self, args, backbone, trans_encoder):
        super(TokenFusion, self).__init__()
        self.radar_backbone = backbone['radar']
        self.image_backbone = backbone['image']
        self.depth_backbone = backbone['depth']
        self.image_trans_encoder = Transformer2(dim=2048, depth=3, heads=8, dim_head=256, mlp_dim=4096)
        self.point_trans_encoder = copy.deepcopy(self.image_trans_encoder)
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.radar_feat_num_dim = torch.nn.Linear(args.num_clusters, 1)
        self.image_feat_dim = torch.nn.Linear(1024, 2048)
        self.depth_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_embedding = torch.nn.Linear(3, 2048)
        self.global_embedding = torch.nn.Embedding(5, 2048)
        self.local_embedding = torch.nn.Embedding(5, 2048)
        self.score_net = torch.nn.ModuleList([PredictorLG(2048) for _ in range(3)])
        self.patch_mlp = torch.nn.ModuleList([FFN(2048, 4096, 2048, 3) for _ in range(3)])
        
        self.input_indices = []
        
    def get_img_patch_idx(self, xyz, trans_mat, cam, bbox):
        proj_uv = project_pcl_torch(xyz, trans_mat, cam=cam)
        bbox_size = bbox[:,2:] - bbox[:,:2]
        trans_uv = (proj_uv - bbox[:,None,:2])*224/bbox_size[:,None]
        pos_idx = (trans_uv // 32).long()
        pos_idx = torch.where(pos_idx < 6, pos_idx, 6)
        pos_idx = torch.where(pos_idx > 0, pos_idx, 0)
        patch_idx = pos_idx[:,:,0] * 7 + pos_idx[:,:,1]
        return patch_idx.unsqueeze(-1).expand(-1,-1,2048)
    
    def process_image(self, images, pos_emb):
        # extract grid features and global image features using a CNN backbone
        _, grid_feat = self.image_backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.image_feat_dim(grid_feat)
        # add embeddings
        embeddings = torch.full((images.shape[0], 1), pos_emb, dtype=torch.long, device=images.device)
        local_pos_emb = self.local_embedding(embeddings)
        grid_feat += local_pos_emb
        
        return grid_feat

    def process_point(self, points, pos_emb):
        # extract cluster features and global point features using a PointNet backbone
        if points.shape[1] == 1024:
            xyz, cluster_feat = self.radar_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
        else:
            xyz, cluster_feat = self.depth_backbone(xyz=points[:,:,:3].contiguous(),features=points[:,:,3:].transpose(1,2).contiguous())
        # add embeddings
        xyz_embedding = self.point_embedding(xyz)
        cluster_feat = cluster_feat.transpose(1,2).contiguous() + xyz_embedding
        embeddings = torch.full((points.shape[0], 1), pos_emb, dtype=torch.long, device=points.device)
        local_pos_emb = self.local_embedding(embeddings)
        cluster_feat += local_pos_emb
        
        return cluster_feat, xyz
    
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
                data_dict[inputs[i]] = torch.zeros_like(data_dict[inputs[i]])
                inputs[i] = ''
                
            # Modality Masking Module
            if not args.wo_MMM:
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
        else:
            mask_inputs = set(args.inputs) - set(args.enabled_inputs)
            for input in mask_inputs:
                data_dict[input] = torch.zeros_like(data_dict[input])
        
        image_feats = []
        point_feats = []
        proj_patch_idx = []
        cluster_centers = {}
        
        # extract local features
        for i, input in enumerate(['radar','master_image','master_depth','sub_image','sub_depth']):
            if data_dict[input].shape[1]:
                if i == 1 or i == 3:
                    image_feats.append(self.process_image(data_dict[input], pos_emb=i))
                else:
                    point_feat, xyz = self.process_point(data_dict[input], pos_emb=i)
                    point_feats.append(point_feat)
                    mas_patch_idx = self.get_img_patch_idx(xyz+data_dict['root_pelvis'][:,None,:].cuda(), 
                                                           data_dict['trans_mat']['master_image'].cuda(), 
                                                           'master', data_dict['bbox']['master_image'].cuda())
                    sub_patch_idx = self.get_img_patch_idx(xyz+data_dict['root_pelvis'][:,None,:].cuda(), 
                                                           data_dict['trans_mat']['sub_image'].cuda(), 
                                                           'sub', data_dict['bbox']['sub_image'].cuda())
                    proj_patch_idx.append([mas_patch_idx, sub_patch_idx])
        
        pred_score = []
        # token fusion
        for i in range(3):
            image_tokens = list(map(self.image_trans_encoder.layers[i], image_feats))
            point_tokens = list(map(self.point_trans_encoder.layers[i], point_feats))
            # score the tokens
            token_scores = torch.nn.functional.softmax(self.score_net[i](torch.cat(image_tokens+point_tokens, dim=1)), -1)[:,:,0]
            pred_score.append(token_scores)
            token_masks = torch.where(token_scores > 0.02, 1, 0).unsqueeze(-1)
            image_token_masks = torch.split(token_masks[:,:49*2], 49, 1)
            point_token_masks = torch.split(token_masks[:,49*2:], 49, 1)
            # replace unimportant tokens
            for j in range(len(image_tokens)):
                image_feats[j] = image_tokens[j] * image_token_masks[j] + image_tokens[1-j] * ~image_token_masks[j]
            for k in range(len(point_tokens)):
                mas_img_token = torch.gather(image_tokens[0], 1, proj_patch_idx[k][0])
                sub_img_token = torch.gather(image_tokens[1], 1, proj_patch_idx[k][1])
                patch_feat = self.patch_mlp[i](mas_img_token) + self.patch_mlp[i](sub_img_token)
                point_feats[k] = point_tokens[k] * point_token_masks[k] + patch_feat * ~point_token_masks[k]

        pred_score = torch.cat(pred_score, dim=1).squeeze()
        global_feats = torch.max(torch.cat(image_feats+point_feats, 1), dim=1, keepdim=True, out=None)[0]
        global_feats = global_feats.expand(-1, ref_vertices.shape[-2], -1)
        # concatinate global features and 3d mesh template
        features = torch.cat([ref_vertices, global_feats], dim=2)

        if is_train:
            # vertex/joint mask
            num_mask = 677
            special_token = torch.ones_like(features[:,:num_mask,:]).cuda()*0.01
            features[:,:num_mask,:] = features[:,:num_mask,:]*meta_masks[:,:num_mask,:] + special_token*(1-meta_masks[:,:num_mask,:])

        pred_dict = {}
        # forward pass
        pred_dict['pred_score'] = pred_score
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