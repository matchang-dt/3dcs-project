import torch
import torch.nn as nn
from dataclasses import dataclass

from .convex_adapter import ConvexAdapterCfg, ConvexAdapter
from .convex_splats import ConvexSplats


@dataclass
class ConvexHeadConfig:
    channels: int          # number of input channels (features + RGB + depth)
    opacity_start: float   # curriculum opacity exponent at step 0
    opacity_end: float     # curriculum opacity exponent at full warmup
    opacity_warmup: float  # number of steps to reach full opacity
    convex_adapter_config: ConvexAdapterCfg


class ConvexHead(nn.Module):
    """
    Predicts convex splat parameters from depth maps, images, and CNN features.

    Analogous to GaussianHead.  The main differences are:
      - outputs K vertex offsets per pixel instead of (scale, quaternion)
      - outputs log_delta and log_sigma instead of per-axis scales / rotation
    The adapter (ConvexAdapter) then converts these predictions into world-space
    ConvexSplats.

    Head output layout per pixel:
      [xy_offset (2) | vertex_offsets (K*3) | log_delta (1) | log_sigma (1) | sh (3*sh_dim)]
    """

    def __init__(self, cfg: ConvexHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.sh_dim = (cfg.convex_adapter_config.sh_degree + 1) ** 2
        K = cfg.convex_adapter_config.nb_points
        channels = cfg.channels

        out_channels = 2 + K * 3 + 1 + 1 + 3 * self.sh_dim # (xy offset, xyz vertices, smoothness, sharpness, shs)
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.adapter = ConvexAdapter(cfg.convex_adapter_config)

    def _map_conf_to_opacity(self, depth_conf: torch.Tensor, global_step: int = 0):
        """Map depth confidence to opacity using the same curriculum as GaussianHead."""
        power = self.cfg.opacity_start + min(
            global_step / self.cfg.opacity_warmup, 1.0
        ) * (self.cfg.opacity_end - self.cfg.opacity_start)
        exp = 2 ** power
        return 0.5 * (1.0 - (1.0 - depth_conf) ** exp + depth_conf ** (1.0 / exp))

    def _get_pixel_centers(
        self, img_shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        H, W = img_shape
        centers_y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
        centers_x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
        grid_y, grid_x = torch.meshgrid(centers_y, centers_x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

    def forward(
        self,
        depth_map: torch.Tensor,   # (B, V, H, W)
        depth_conf: torch.Tensor,  # (B, V, H, W)
        images: torch.Tensor,      # (B, V, 3, H, W)
        features: torch.Tensor,    # (B, V, H//4, W//4, C)
        extrinsics: torch.Tensor,  # (B, V, 4, 4)
        intrinsics: torch.Tensor,  # (B, V, 3, 3)
        global_step: int = 0,
    ) -> ConvexSplats:
        B, V, H, W = depth_map.shape
        device = depth_map.device
        _, _, h, w, C = features.shape

        # depth confidence -> opacities
        opacities = self._map_conf_to_opacity(depth_conf, global_step)

        # Upsample features to full image resolution
        features_flat = features.view(B * V, h, w, C).permute(0, 3, 1, 2)
        features_up = nn.functional.interpolate(
            features_flat, size=(H, W), mode="bilinear", align_corners=False
        )  # (B*V, C, H, W)

        # Build head input: depth(1) + RGB(3) + features(C)
        depth_flat  = depth_map.view(B * V, 1, H, W)
        images_flat = images.view(B * V, 3, H, W)
        head_input  = torch.cat([depth_flat, images_flat, features_up], dim=1)

        pre_splats = self.head(head_input)                   # (B*V, out_channels, H, W)
        pre_splats = pre_splats.permute(0, 2, 3, 1)          # (B*V, H, W, out_channels)
        pre_splats = pre_splats.view(B, V, H, W, -1)         # (B, V, H, W, out_channels)

        # Compute sub-pixel centres using the predicted xy offsets
        pixel_centers = self._get_pixel_centers((H, W), device)  # (H, W, 2)
        pixel_centers = pixel_centers[None, None].expand(B, V, -1, -1, -1)  # (B, V, H, W, 2)
        xy_offsets  = torch.sigmoid(pre_splats[..., :2]) - 0.5 # maybe we don't need this anymore, but will stay consistent
        pixel_size  = torch.tensor([1.0 / W, 1.0 / H], device=device)
        pixel_centers = pixel_centers + xy_offsets * pixel_size

        # Flatten spatial dims; add surfaces dim for adapter
        rays = H * W
        srf  = self.cfg.convex_adapter_config.num_surfaces
        pre_splats_flat    = pre_splats[..., 2:].reshape(B, V, rays, srf, -1)  # (B, V, R, srf, C-2)
        pixel_centers_flat = pixel_centers.reshape(B, V, rays, srf, 2)
        opacities_flat     = opacities.reshape(B, V, rays)
        depths_flat        = depth_map.reshape(B, V, rays)

        return self.adapter(
            pre_splats=pre_splats_flat,
            pixel_centers=pixel_centers_flat,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            opacities=opacities_flat,
            depths=depths_flat,
            img_shape=(H, W),
        )
