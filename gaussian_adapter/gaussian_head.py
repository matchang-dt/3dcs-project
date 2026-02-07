import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from gaussian_adapter.gaussian_adapter import GaussianAdapterCfg, GaussianAdapter


@dataclass
class GaussianHeadConfig:
    channels: int # for density >> opacity
    opacity_start: float # start of training exponent
    opacity_end: float # end exponent
    opacity_warmup: float # how many steps to warmup
    gaussian_adapter_config: GaussianAdapterCfg


class GaussianHead(nn.Module):
    """
    Predicts gaussian parameters. Specifically, the opacity, covariance, and colors that require NN eval.
    """
    def __init__(self, cfg: GaussianHeadConfig):
        super().__init__()
        self.cfg = cfg
        # head for xy offsets, rotation (quaternion rep.), scales, sh coefficients 
        # [xy-offset(2), scales(3), rotation(4), sh(3*sh_dim)]
        self.sh_dim = (self.cfg.gaussian_adapter_config.sh_degree + 1) ** 2
        channels = self.cfg.channels
        out_channels = 2 + 3 + 4 + 3 * self.sh_dim  # xy-offset + scales + rotation + sh
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),  # input: channels=UNet features, RGB, upsampled features (from cnn + tf)
            nn.GELU(),
            nn.Conv2d(channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.adapter = GaussianAdapter(self.cfg.gaussian_adapter_config)

    def _map_conf_to_opacity(self, depth_conf, global_step=0):
        # depth_conf: [B, V, H, W] (already softmaxed->max along depth channels)
        # apparently, global step is used for curriculum learning for training stability
        # opacities: [B, V, H, W]
        power = self.cfg.opacity_start + min(global_step / self.cfg.opacity_warmup, 1) * (self.cfg.opacity_end - self.cfg.opacity_start)
        exp = 2 ** power
        opacities = 1/2 * (1 - (1 - depth_conf) ** exp + depth_conf ** (1/exp))
        return opacities

    def _get_pixel_centers(self, img_shape):
        """Get center pixel normalized coordinates for each pixel in the image."""
        H, W = img_shape
        # Create normalized pixel coordinates [0, 1]
        centers_y = torch.linspace(0.5 / H, 1 - 0.5 / H, H)
        centers_x = torch.linspace(0.5 / W, 1 - 0.5 / W, W)
        grid_y, grid_x = torch.meshgrid(centers_y, centers_x, indexing='ij')
        centers = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        return centers


    def forward(self, depth_map, depth_conf, images, features, extrinsics, intrinsics, global_step=0):
        # depth_map: [B, V, H, W]
        # depth_conf: [B, V, H, W]
        # images: [B, V, 3, H, W]
        # features: [B, V, H//4, W//4, 128]
        # extrinsics: [B, V, 4, 4]
        # intrinsics: [B, V, 3, 3]
        B, V, H, W = depth_map.shape
        device = depth_map.device
        _, _, h, w, C = features.shape  # h = H//4, w = W//4
        
        # Map confidence to opacity
        opacities = self._map_conf_to_opacity(depth_conf, global_step)
        
        # Upsample features to full resolution
        features_flat = features.view(B * V, h, w, C).permute(0, 3, 1, 2)  # [B*V, 128, H//4, W//4]
        features_upsampled = torch.nn.functional.interpolate(
            features_flat, size=(H, W), mode='bilinear', align_corners=False
        )  # [B*V, 128, H, W]
        
        # Predict gaussian params
        depth_map_flat = depth_map.view(B * V, 1, H, W)
        images_flat = images.view(B * V, 3, H, W)
        ghead_input = torch.cat([depth_map_flat, images_flat, features_upsampled], dim=1)  # [B*V, 132, H, W]
        pre_gaussians = self.head(ghead_input)  # [B*V, C, H, W]
        pre_gaussians = pre_gaussians.permute(0, 2, 3, 1)  # [B*V, H, W, C]
        pre_gaussians = pre_gaussians.view(B, V, H, W, -1)  # [B, V, H, W, C]
        
        # Get pixel centers and offsets for subpixel preds
        pixel_centers = self._get_pixel_centers(img_shape=(H, W)).to(device)  # [H, W, 2]
        pixel_centers = pixel_centers.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1, -1)  # [B, V, H, W, 2]
        xy_offsets = torch.sigmoid(pre_gaussians[..., :2]) - 0.5  # [-0.5, 0.5]
        pixel_size = torch.tensor([1.0 / W, 1.0 / H], device=device)
        pixel_centers = pixel_centers + xy_offsets * pixel_size
        
        # Flatten spatial dimensions and add srf dimension for adapter
        rays = H * W
        srf = self.cfg.gaussian_adapter_config.num_surfaces  # num_surfaces
        pre_gaussians_flat = pre_gaussians[..., 2:].reshape(B, V, rays, srf, -1) # [B, V, rays, srf, C-2]
        pixel_centers_flat = pixel_centers.reshape(B, V, rays, srf, 2)  # [B, V, rays, srf, 2]
        opacities_flat = opacities.reshape(B, V, rays)  # [B, V, rays]
        depths_flat = depth_map.reshape(B, V, rays)  # [B, V, rays]
        
        gaussians = self.adapter(
            pre_gaussians=pre_gaussians_flat,
            pixel_centers=pixel_centers_flat,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            opacities=opacities_flat,
            depths=depths_flat,
            img_shape=(H, W),
        )
        return gaussians

# unproject depth to get 3D points (use as means)

# depth_conf (already in max softmax) opacity -- this is part of the depth map prediction in the original model (2-dim per pixel for depth, conf)
# (was calculated using 1/2(1 - (1 - x)^t + x^{1/t})) that maps prob density [0,1] to opacity values (see line 380
# - raw density, used sigmoid to get into [B,V,R=H*W, num_surfaces, gpp]

# in depth pred, already predicts scales, rotations, sh used to build the covariance matrix + color (need to have separate heads in here for this)
