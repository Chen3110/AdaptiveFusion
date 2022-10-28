"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import random
import torch
import numpy as np
from src.datasets.utils import gen_mask_indices
from src.modeling.bert.transformer import Transformer


class FusionGraphormer(torch.nn.Module):
    def __init__(self, backbone, trans_encoder):
        super(FusionGraphormer, self).__init__()
        self.point_backbone = backbone['point']
        self.image_backbone = backbone['image']
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(655, 2619)
        self.upsampling2 = torch.nn.Linear(2619, 10475)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)
        self.image_feat_dim = torch.nn.Linear(2048, 1024)
        
        self.point_feat_num_dim = torch.nn.Linear(32, 1)
        # self.point_feat_num_dim = torch.nn.Linear(49, 1)
        self.point_feat_dim = torch.nn.Linear(2048, 1024)
        self.point_embedding = torch.nn.Linear(3, 2051)
        
        self.GIM = Transformer(dim=2048, depth=3, heads=8, dim_head=256, mlp_dim=4096)

    def process_image(self, images):
        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.image_backbone(images)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1,2)
        grid_feat = self.grid_feat_dim(grid_feat)
        return image_feat.view(images.shape[0],1,2048), grid_feat

    def process_point(self, points):
        # extract cluster features and global point features using a PointNet backbone
        xyz, cluster_feat = self.point_backbone(xyz=points[:,:,:3].contiguous(), features=points[:,:,3:].transpose(1,2).contiguous())
        xyz_embedding = self.point_embedding(xyz)
        point_feat = self.point_feat_num_dim(cluster_feat).squeeze()
        cluster_feat = torch.cat([xyz, cluster_feat.transpose(1,2).contiguous()], dim=2)
        cluster_feat = xyz_embedding + cluster_feat
        return point_feat.view(points.shape[0],1,2048), cluster_feat
    
    def forward(self, args, data_dict, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = data_dict['joints_3d'].shape[0]
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,69))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda(args.device)
        template_betas = torch.zeros((1,16)).cuda(args.device)
        template_res = smpl(template_pose, template_betas)
        template_vertices = template_res['verts']

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = template_res['joints'][:, :22]
        template_pelvis = template_3d_joints[:,0,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        if is_train:
            # image mask
            image_mask = np.ones((batch_size, 1, 1, 1))
            image_indices = gen_mask_indices(batch_size, args.mask_ratio)
            image_mask[image_indices] = 0.0
            image_mask = torch.from_numpy(image_mask).float().to(args.device)
            image_mask = image_mask.expand(-1, 3, 224, 224)
            data_dict['master_image'] *= image_mask
            data_dict['sub_image'] *= image_mask
            
            # modality mask
            modality_indices = gen_mask_indices(max_mask_num=len(args.resource), mask_ratio=1.)
            for i in modality_indices:
                data_dict[args.resource[i]] = torch.Tensor([[]])
        
        global_feats = []
        local_feats = []
        
        if data_dict['radar'].shape[1]:
            radar_gloabl_feat, radar_local_feat = self.process_point(data_dict['radar'])
            global_feats.append(radar_gloabl_feat)
            local_feats.append(radar_local_feat)
            
        if data_dict['master_image'].shape[1]:
            mas_img_gloabl_feat, mas_img_local_feat = self.process_image(data_dict['master_image'])
            global_feats.append(mas_img_gloabl_feat)
            local_feats.append(mas_img_local_feat)
            
        if data_dict['master_depth'].shape[1]:
            mas_depth_gloabl_feat, mas_depth_local_feat = self.process_point(data_dict['master_depth'])
            global_feats.append(mas_depth_gloabl_feat)
            local_feats.append(mas_depth_local_feat)
            
        if data_dict['sub_image'].shape[1]:
            sub_img_gloabl_feat, sub_img_local_feat = self.process_image(data_dict['sub_image'])
            global_feats.append(sub_img_gloabl_feat)
            local_feats.append(sub_img_local_feat)
            
        if data_dict['sub_depth'].shape[1]:
            sub_depth_gloabl_feat, sub_depth_local_feat = self.process_point(data_dict['sub_depth'])
            global_feats.append(sub_depth_gloabl_feat)
            local_feats.append(sub_depth_local_feat)
            
        if is_train:
            # shuffle global and local features
            feat_zip = list(zip(global_feats, local_feats))
            random.shuffle(feat_zip)
            global_feats, local_feats = zip(*feat_zip)
        # concatinate global and local features
        global_feats = torch.cat(global_feats, dim=1)
        local_feats = torch.cat(local_feats, dim=1)
        # integrate image and point global features
        fusion_feat = self.GIM(global_feats)
        fusion_feat = torch.sum(fusion_feat, dim=1, keepdim=True).expand(-1,ref_vertices.shape[-2], -1)
        # concatinate image/point feat and 3d mesh template
        features = torch.cat([ref_vertices, fusion_feat], dim=2)
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
            pred_dict['attention'] = att
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
