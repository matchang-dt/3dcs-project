import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from .refiner import DepthRefiner


def depth_estimate(cost_volume, max_depth):
    # cost volume: [B, K, H, W, 128]
    b, k, H, W, d = cost_volume.shape
    depth_prob = torch.softmax(cost_volume, dim=-1)
    depth_invs = torch.arange(d, device=cost_volume.device) + 1
    depths = (max_depth / depth_invs).reshape(1, 1, 1, 1, d).expand(b, k, H, W, d) # [B, K, H, W, d (128)]
    depth_map = torch.einsum('bkHWd,bkHWd->bkHW', depth_prob, depths) # [B, K, H, W]
    return depth_map # [B, K, H, W]


class DepthEstimator(L.LightningModule):
    def __init__(self, max_depth,channels=128, feat_map_size=256, dtype=torch.float32):
        super().__init__()
        self.to(dtype)
        self.max_depth = max_depth
        self.refiner = DepthRefiner(channels, feat_map_size, dtype)
    
    def forward(self, cost_volume, images, features):
        # cost_volume: [B, K, H, W, 128]
        # images: [B, K, 3, H, W]
        # features: [B, K, H//4, W//4, 128]
        b, k, H, W, d = cost_volume.shape # h=H//4, w=W//4
        images = images.reshape(-1, 3, H, W) # [B*K, 3, H, W]
        features = features.reshape(-1, H//4, W//4, d).permute(0, 3, 1, 2) # [B*K, 128, H//4, W//4]
        features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False) # [B*K, 128, H, W]
        depth_map = depth_estimate(cost_volume, self.max_depth) # [B, K, H, W]
        depth_map_usq = depth_map.reshape(-1, H, W).unsqueeze(1) # [B*K, 1, H, W]
        
        refine_inputs = torch.cat([images, features, depth_map_usq], dim=1) # [B*K, 128+4, H, W]
        refine_inputs = refine_inputs.permute(0, 2, 3, 1).reshape(b, k, H, W, d+4) # [B*K, H, W, 128+4]
        depth_residual = self.refiner(refine_inputs) # [B, K, H, W]
        depth_map += depth_residual # [B, K, H, W]
        return depth_map # [B, K, H, W]
