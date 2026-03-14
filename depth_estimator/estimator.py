import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from .refiner import DepthRefiner


def inv_depth_estimate(cost_volume, near, far):
    """
    Estimate the depth map from the cost volume.
    Args:
        cost_volume (torch.Tensor): input tensor of shape [B, K, H, W, D=128]
        near (float): near plane
        far (float): far plane
    Returns:
        depth_map (torch.Tensor): output tensor of shape [B, K, H, W]
        depth_conf (torch.Tensor): output tensor of shape [B, K, H, W], the max probability of the depth candidate for each pixel
    """
    # cost volume: [B, K, H, W, 128]
    b, k, H, W, d = cost_volume.shape
    depth_prob = torch.softmax(cost_volume, dim=-1) # [B, K, H, W, d (128)]
    inv_depths = torch.linspace(1/far, 1/near, d, dtype=cost_volume.dtype, device=cost_volume.device)
    inv_depths = inv_depths.reshape(1, 1, 1, 1, d).expand(b, k, H, W, d) # [B, K, H, W, d (128)]
    inv_depth_map = torch.einsum('bkHWd,bkHWd->bkHW', depth_prob, inv_depths) # [B, K, H, W]
    depth_conf = torch.max(depth_prob, dim=-1)[0]
    return inv_depth_map, depth_conf # [B, K, H, W], [B, K, H, W] (for gaussian mean, opacity)


class DepthEstimator(L.LightningModule):
    """
    Depth estimator module.
    Estimates the depth map from the cost volume, then refines the depth map with the features and the images by a U-net based refiner.
    """
    def __init__(self, near, far, channels=128, feat_map_size=256, dtype=torch.float32):
        """
        Initialize the DepthEstimator.
        Args:
            near (float): near plane
            far (float): far plane
            channels (int): number of channels for the features
            feat_map_size (int): size of the feature map
            dtype (torch.dtype): data type
        """
        super().__init__()
        num_groups = channels // 16
        self.to(dtype)
        self.near = near
        self.far = far
        self.refiner = DepthRefiner(channels, feat_map_size, dtype)
        self.small_depth_head = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype),
        ) # for raw cost volume
    
    def forward(self, cost_volume, images, features):
        """
        Forward pass of the DepthEstimator.
        Args:
            cost_volume (torch.Tensor): input tensor of shape [B, K, H, W, D=128]
            images (torch.Tensor): input tensor of shape [B, K, 3, H, W]
            features (torch.Tensor): input tensor of shape [B, K, H, W, 128] features extracted from the extractor
        Returns:
            depth_map (torch.Tensor): output tensor of shape [B, K, H, W]
            depth_conf (torch.Tensor): output tensor of shape [B, K, H, W], the max probability of the depth candidate for each pixel
        """
        b, k, H, W, d = cost_volume.shape # h=H//4, w=W//4
        images = images.reshape(-1, 3, H * 4, W * 4) # [B*K, 3, H, W]
        cost_volume = self.small_depth_head(cost_volume.reshape(b*k, H, W, d).permute(0,3,1,2)) # [B, K, H, W, 128]
        cost_volume = cost_volume.permute(0, 2, 3, 1).reshape(b, k, H, W, d) # [B, K, H, W, 128]
        # cost volume here should be downsampled still
        inv_depth_map, depth_conf = inv_depth_estimate(cost_volume, self.near, self.far) # [B, K, H, W], [B, K, H, W]
        # upsample
        inv_depth_map = F.interpolate(inv_depth_map, size=(H*4, W*4), mode='bilinear', align_corners=False)
        depth_conf = F.interpolate(depth_conf, size=(H*4, W*4), mode='bilinear', align_corners=False)

        inv_depth_map_usq = inv_depth_map.reshape(-1, H * 4, W * 4).unsqueeze(1) # [B*K, 1, H, W]
        depth_conf_usq = depth_conf.reshape(-1, H * 4, W * 4).unsqueeze(1) # [B*K, 1, H, W]

        # (features are already upsampled outside, or they should be)
        refine_inputs = torch.cat([images, features, inv_depth_map_usq, depth_conf_usq], dim=1) # [B*K, 128+5, H, W]
        refine_inputs = refine_inputs.permute(0, 2, 3, 1).reshape(b, k, H*4, W*4, d+5) # [B*K, H, W, 128+5]
        inv_depth_residual = self.refiner(refine_inputs) # [B, K, H, W]
        # print("================================================")
        # print("inv_depth_map min, max: ", inv_depth_map.min(), inv_depth_map.max())
        # print("inv_depth_map mean, std: ", inv_depth_map.mean(), inv_depth_map.std())
        # print("inv_depth_residual min, max: ", inv_depth_residual.min(), inv_depth_residual.max())
        # print("inv_depth_residual mean, std: ", inv_depth_residual.mean(), inv_depth_residual.std())
        inv_depth_map += inv_depth_residual # [B, K, H, W]
        inv_depth_map = inv_depth_map.clamp(max=1/self.near, min=1/self.far)
        depth_map = 1 / inv_depth_map # [B, K, H, W]
        return depth_map, depth_conf # [B, K, H, W], [B, K, H, W]
