import torch
import torch.nn as nn
from openpoints.models.build import build_model_from_cfg
from openpoints.utils.config import EasyConfig

class PointNeXt(nn.Module):
    def __init__(self, cfg_file=None):
        super().__init__()
        ecfg = EasyConfig()
        if cfg_file is None:
            ecfg.load('cfgs/pointnext-s.yaml')
        else:
            ecfg.load(cfg_file)
        self.encoder = build_model_from_cfg(ecfg.model)
        
    def forward(self, xyz):
        points, feats = self.encoder(xyz)
        return points, feats


if __name__ == "__main__":
    model = PointNeXt().cuda()
    points = torch.rand(10, 1024, 3).cuda()
    out = model(points)
    print(out)