import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from src.modeling._smpl import SMPLX, SMPLX_MODEL_NEUTRAL_PATH


class GeodesicLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(GeodesicLoss, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            # breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta


class MeshLoss(_Loss):
    def __init__(self, device=torch.device('cpu'), size_average=None, reduce=None, reduction: str = 'mean', scale: float = 1) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.smplx_model = SMPLX(bm_fname=SMPLX_MODEL_NEUTRAL_PATH).to(device)
        self.scale = scale

    def forward(self, input: torch.Tensor, target: torch.Tensor, use_gender: bool = False, train: bool = True) -> torch.Tensor:

        _input = input * self.scale
        _target = target * self.scale

        if not use_gender:
            input_model = target_model = self.smplx_model
        else:
            raise NotImplementedError()

        input_params = dict(
            pose_body=_input[:, :-16],
            betas=_input[:, -16:],
        )

        input_result = input_model(use_rodrigues=False, **input_params)
        input_verts = input_result['vertices']
        input_joints = input_result['joints']

        target_params = dict(
            pose_body=_target[:, :-16],
            betas=_target[:, -16:],
        )

        target_result = target_model(use_rodrigues=False, **target_params)
        target_verts = target_result['vertices']
        target_joints = target_result['joints']
        
        per_joint_err = torch.norm((input_joints - target_joints), dim=-1)
        per_vertex_err = torch.norm((input_verts - target_verts), dim=-1)

        if train:
            return (F.l1_loss(input_verts, target_verts, reduction=self.reduction), 
                    F.l1_loss(input_joints, target_joints, reduction=self.reduction))
        else:
            return (torch.sqrt(F.mse_loss(input_verts, target_verts, reduction=self.reduction)),
                    torch.sqrt(F.mse_loss(input_joints, target_joints, reduction=self.reduction)),
                    (per_joint_err, per_vertex_err))
