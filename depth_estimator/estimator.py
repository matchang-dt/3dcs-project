import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from .refiner import DepthRefiner


def depth_estimate(cost_volume, max_depth):
    """
    Estimate the depth map from the cost volume.
    Args:
        cost_volume (torch.Tensor): input tensor of shape [B, K, H, W, D=128]
        max_depth (float): maximum depth (far plane)
    Returns:
        depth_map (torch.Tensor): output tensor of shape [B, K, H, W]
        depth_conf (torch.Tensor): output tensor of shape [B, K, H, W], the max probability of the depth candidate for each pixel
    """
    # cost volume: [B, K, H, W, 128]
    b, k, H, W, d = cost_volume.shape
    depth_prob = torch.softmax(cost_volume, dim=-1)
    depth_invs = torch.arange(d, device=cost_volume.device) + 1
    depths = (max_depth / depth_invs).reshape(1, 1, 1, 1, d).expand(b, k, H, W, d) # [B, K, H, W, d (128)]
    depth_map = torch.einsum('bkHWd,bkHWd->bkHW', depth_prob, depths) # [B, K, H, W]
    depth_conf = torch.max(depth_prob, dim=-1)[0]
    return depth_map, depth_conf # [B, K, H, W], [B, K, H, W] (for gaussian mean, opacity)


class DepthEstimator(L.LightningModule):
    """
    Depth estimator module.
    Estimates the depth map from the cost volume, then refines the depth map with the features and the images by a U-net based refiner.
    """
    def __init__(self, max_depth, channels=128, feat_map_size=256, dtype=torch.float32):
        """
        Initialize the DepthEstimator.
        Args:
            max_depth (float): maximum depth (far plane)
            channels (int): number of channels for the features
            feat_map_size (int): size of the feature map
            dtype (torch.dtype): data type
        """
        super().__init__()
        self.to(dtype)
        self.max_depth = max_depth
        self.refiner = DepthRefiner(channels, feat_map_size, dtype)
    
    def forward(self, cost_volume, images, features):
        """
        Forward pass of the DepthEstimator.
        Args:
            cost_volume (torch.Tensor): input tensor of shape [B, K, H, W, D=128]
            images (torch.Tensor): input tensor of shape [B, K, 3, H, W]
            features (torch.Tensor): input tensor of shape [B, K, H//4, W//4, 128] features extracted from the extractor
        Returns:
            depth_map (torch.Tensor): output tensor of shape [B, K, H, W]
            depth_conf (torch.Tensor): output tensor of shape [B, K, H, W], the max probability of the depth candidate for each pixel
        """
        b, k, H, W, d = cost_volume.shape # h=H//4, w=W//4
        images = images.reshape(-1, 3, H, W) # [B*K, 3, H, W]
        features = features.reshape(-1, H//4, W//4, d).permute(0, 3, 1, 2) # [B*K, 128, H//4, W//4]
        features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False) # [B*K, 128, H, W]
        depth_map, depth_conf = depth_estimate(cost_volume, self.max_depth) # [B, K, H, W], [B, K, H, W, d (128)]
        depth_map_usq = depth_map.reshape(-1, H, W).unsqueeze(1) # [B*K, 1, H, W]
        
        refine_inputs = torch.cat([images, features, depth_map_usq], dim=1) # [B*K, 128+4, H, W]
        refine_inputs = refine_inputs.permute(0, 2, 3, 1).reshape(b, k, H, W, d+4) # [B*K, H, W, 128+4]
        depth_residual = self.refiner(refine_inputs) # [B, K, H, W]
        depth_map += depth_residual # [B, K, H, W]
        return depth_map, depth_conf # [B, K, H, W], [B, K, H, W]
