import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GaussianHeadConfig:
    input_dim: int
    opacity_head_channels: int # for density >> opacity
    transf_head_channels: int  # for covariance, color info
    opacity_t: float # tuner for opacity mapping
    sh_degree: int   # spherical harmonics degree
    scale_min: float = 0.01
    scale_max: float = 100.
    num_surfaces: int  = 1 # NOTE: currently only works with '1'
    gaussians_per_pixel: int = 1 # NOTE: currently only works with '1'
    gaussian_scale_pct: int = 1


class GaussianHead(nn.Module):
    """
    Predicts gaussian parameters. Specifically, the opacity, covariance, and colors that require NN eval.
    """
    def __init__(self, cfg: GaussianHeadConfig):
        super().__init__(cfg)
        # head for rotation (quaternion rep.), scales, sh coefficients 
        # num params here are num_surfaces * (d_in * 2) -> 84(xy,scales=3,quat=4,sh=[sh_deg=4 + 1] ** 2 * 3)
        # IN: exact the same as the depthest out dim channels
        self.sh_dim = (self.cfg.sh_degree + 1) ** 2
        channels = self.cfg.channels
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
        )

    def _map_conf_to_opacity(self, depth_conf):
        # depth_conf: [B, V, H, W] (post softmax->max along depth channels)
        # out: [B, V, H, W]
        t = self.cfg.opacity_t # some config value
        return 1/2 * (1 - (1 - depth_conf) ** t + depth_conf ** (1/t))

    def _unproject_depth_to_points(self, depth_map, scales):
        # depth_map: [B, V, H, W]
        # output: gaussian means (3D points) based on depth
        pass

    def forward(self, depth_map, depth_conf, images, extrinsics, intrinsics): # input to this should be the depth maps, conf maps, images, upsampled features
        # depth_map: [B, V, H, W]
        # depth_: [B, V, H, W]
        # images: [B, V, 3, H, W]
        # features: [B, V, H//4, W//4, 128]
        # these should be concat along V
        # depth_* come from the depth estimator post-refinement
        opacities = self._map_conf_to_opacity(depth_conf)
        pre_gaussians = self.head(depth_map)
        scales, quaterinons, sh_coefs = torch.split(pre_gaussians, (3,4,3 * self.sh_dim), dim=-1)
        quaterinons = F.normalize(quaterinons, dim=-1)

        means = self._unproject_depth_to_points(depth_map, scales)
        return means, opacities, scales, quaterinons, sh_coefs

# unproject depth to get 3D points (use as means)

# depth_conf (already in max softmax) opacity -- this is part of the depth map prediction in the original model (2-dim per pixel for depth, conf)
# (was calculated using 1/2(1 - (1 - x)^t + x^{1/t})) that maps prob density [0,1] to opacity values (see line 380
# - raw density, used sigmoid to get into [B,V,R=H*W, num_surfaces, gpp]

# in depth pred, already predicts scales, rotations, sh used to build the covariance matrix + color (need to have separate heads in here for this)
