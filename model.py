import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union

from extractor.extractor import Extractor
from cost_volume_constructor.constructor import CostVolumeConstructor
from depth_estimator.estimator import DepthEstimator
from gaussian_adapter.gaussian_head import GaussianHead, GaussianHeadConfig
from gaussian_adapter.gaussian_adapter import GaussianAdapterCfg
from convex_adapter.convex_head import ConvexHead, ConvexHeadConfig
from convex_adapter.convex_adapter import ConvexAdapterCfg
from decoder.decoder_cuda_splatting_gaussians import (
    DecoderGaussianSplattingCUDA,
    DecoderGaussianSplattingCUDACfg,
)
from decoder.decoder_cuda_splatting_convexes import (
    DecoderConvexSplattingCUDA,
    DecoderConvexSplattingCUDACfg,
)
from datasets.dataset import DatasetCfg
from utils.projection import make_proj_matrix
import torch.nn.functional as F


@dataclass
class MVSplatConfig:
    """Configuration for MVSplat model."""
    # Extractor params
    image_size: int = 256
    hidden_dim: int = 128
    swin_divisions: int = 2
    cnn_dtype: torch.dtype = torch.float32
    transformer_dtype: torch.dtype = torch.bfloat16

    # Pipeline dtype: used for features after extractor and for cost volume / depth estimator.
    # Set to float32 to match default module weights, or to transformer_dtype to avoid casting.
    pipeline_dtype: torch.dtype = torch.float32

    # Cost volume params
    near: float = 1.0
    far: float = 100.0
    feature_dim: int = 128

    # Splat head params (shared between Gaussian and Convex heads)
    gaussian_head_channels: int = 132  # 128 (features) + 4 (RGB + depth)
    opacity_start: float = 0.5
    opacity_end: float = 2.0
    opacity_warmup: float = 10000

    # Gaussian adapter params
    sh_degree: int = 4
    scale_min: float = 0.5
    scale_max: float = 15.0
    gaussian_scale_pct: float = 0.1
    gaussians_per_pixel: int = 1
    num_surfaces: int = 1

    # Convex splat params (only used when use_convex=True)
    use_convex: bool = False
    nb_points: int = 6           # K: vertices per convex splat (max 8 per CUDA config)
    splat_scale_pct: float = 0.1 # analogous to gaussian_scale_pct

    # Decoder params
    decoder_cfg: Optional[Union[DecoderGaussianSplattingCUDACfg, DecoderConvexSplattingCUDACfg]] = None
    dataset_cfg: Optional[DatasetCfg] = None

    @property
    def head_channels(self) -> int:
        """Alias so both Gaussian and Convex heads use the same field."""
        return self.gaussian_head_channels

class MVSplat(nn.Module):
    """
    MVSplat: Multi-View Splat model for novel view synthesis.

    Pipeline:
    1. Extractor: Extract features from input images
    2. Cost Volume Constructor: Build cost volume from features
    3. Depth Estimator: Estimate depth maps from cost volume
    4. Splat Head: Predict splat parameters (Gaussian or Convex) from depth + features
    5. Decoder: Render splats to images via CUDA rasterizer

    Set cfg.use_convex=True to use convex splats (ConvexHead + DecoderConvexSplattingCUDA)
    instead of the default Gaussian splats.
    """

    def __init__(self, cfg: MVSplatConfig):
        super().__init__()
        self.cfg = cfg

        self.extractor = Extractor(
            image_size=cfg.image_size,
            hidden_dim=cfg.hidden_dim,
            swin_divisions=cfg.swin_divisions,
            cnn_dtype=cfg.cnn_dtype,
            transformer_dtype=cfg.transformer_dtype,
        )

        h = cfg.image_size // 4
        w = cfg.image_size // 4
        self.cost_volume_constructor = CostVolumeConstructor(
            h=h,
            w=w,
            near=cfg.near,
            far=cfg.far,
            feature_dim=cfg.feature_dim,
            dtype=cfg.pipeline_dtype,
        )

        self.conv1 = nn.Conv2d(cfg.feature_dim * 2, cfg.feature_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cfg.feature_dim, cfg.feature_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn = nn.GroupNorm(num_groups=cfg.feature_dim // 16, num_channels=cfg.feature_dim, eps=1e-6)
        self.silu = nn.SiLU(inplace=True)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.gn.weight, 1)
        nn.init.constant_(self.gn.bias, 0)

        # interpolate wrapper (for functional)
        class Interpolate(nn.Module):
            def __init__(self, size, mode='bilinear', align_corners=False):
                super().__init__()
                self.size = size
                self.mode = mode
                self.align_corners = align_corners

            def forward(self, x):
                return F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)

        self.feature_upsampler = nn.Sequential(
            Interpolate(size=(h * 2, w * 2)),
            self.conv1,
            self.gn,
            self.silu,
            Interpolate(size=(h * 4, w * 4)),
            self.conv2
        )

        self.depth_estimator = DepthEstimator(
            near=cfg.near,
            far=cfg.far,
            channels=cfg.feature_dim,
            feat_map_size=cfg.image_size,
            dtype=cfg.pipeline_dtype,
        )

        if cfg.use_convex:
            convex_adapter_cfg = ConvexAdapterCfg(
                sh_degree=cfg.sh_degree,
                nb_points=cfg.nb_points,
                scale_min=cfg.scale_min,
                scale_max=cfg.scale_max,
                splat_scale_pct=cfg.splat_scale_pct,
                splats_per_pixel=cfg.gaussians_per_pixel,
                num_surfaces=cfg.num_surfaces,
            )
            convex_head_cfg = ConvexHeadConfig(
                channels=cfg.gaussian_head_channels,
                opacity_start=cfg.opacity_start,
                opacity_end=cfg.opacity_end,
                opacity_warmup=cfg.opacity_warmup,
                convex_adapter_config=convex_adapter_cfg,
            )
            self.splat_head = ConvexHead(convex_head_cfg)

            if cfg.decoder_cfg is None:
                cfg.decoder_cfg = DecoderConvexSplattingCUDACfg(
                    name="cuda_convex_splatting",
                    sh_degree=cfg.sh_degree,
                )
            self.decoder = DecoderConvexSplattingCUDA(cfg.decoder_cfg, cfg.dataset_cfg)
        else:
            gaussian_adapter_cfg = GaussianAdapterCfg(
                sh_degree=cfg.sh_degree,
                scale_min=cfg.scale_min,
                scale_max=cfg.scale_max,
                gaussian_scale_pct=cfg.gaussian_scale_pct,
                gaussians_per_pixel=cfg.gaussians_per_pixel,
                num_surfaces=cfg.num_surfaces,
            )
            gaussian_head_cfg = GaussianHeadConfig(
                channels=cfg.gaussian_head_channels,
                opacity_start=cfg.opacity_start,
                opacity_end=cfg.opacity_end,
                opacity_warmup=cfg.opacity_warmup,
                gaussian_adapter_config=gaussian_adapter_cfg,
            )
            self.splat_head = GaussianHead(gaussian_head_cfg)

            if cfg.decoder_cfg is None:
                cfg.decoder_cfg = DecoderGaussianSplattingCUDACfg(name="cuda_gaussian_splatting")
            self.decoder = DecoderGaussianSplattingCUDA(cfg.decoder_cfg, cfg.dataset_cfg)

    def forward(self, batch, global_step=0, render_depth=False):
        """
        Forward pass of MVSplat model.

        Args:
            batch: Dictionary with context/target images, intrinsics, extrinsics, near_plane, far_plane.
            global_step: Current training step (for opacity curriculum).
            render_depth: Whether to render depth maps.

        Returns:
            Dict with rendered_images, rendered_depth (optional), depth_maps, depth_conf, gaussians.
        """
        context_images = batch["context"]["images"]
        context_intrinsics = batch["context"]["intrinsics"]
        context_extrinsics = batch["context"]["extrinsics"]

        target_intrinsics = batch["target"]["intrinsics"]
        target_extrinsics = batch["target"]["extrinsics"]

        near_plane = self.cfg.near
        far_plane = self.cfg.far

        B, K, C, H, W = context_images.shape
        num_target_views = target_extrinsics.shape[1]

        if isinstance(near_plane, (int, float)):
            near_plane = torch.tensor(
                [near_plane] * B * num_target_views,
                device=context_images.device,
            ).view(B, num_target_views)
        if isinstance(far_plane, (int, float)):
            far_plane = torch.tensor(
                [far_plane] * B * num_target_views,
                device=context_images.device,
            ).view(B, num_target_views)

        # 1. Extract features
        features, features_cnn = self.extractor(context_images)
        features = features.to(self.cfg.pipeline_dtype) # [B, V, H//4, W//4, 128]
        features_cnn = features_cnn.to(self.cfg.pipeline_dtype) # [B, V, 128, H//4, W//4]

        with torch.autocast(device_type='cuda', enabled=False):
            proj_matrices = make_proj_matrix(context_extrinsics, context_intrinsics)
        proj_matrices = proj_matrices.to(self.cfg.pipeline_dtype)

        # near and far planes should be constant
        n = near_plane.min() if isinstance(near_plane, torch.Tensor) else near_plane
        f = far_plane.max() if isinstance(far_plane, torch.Tensor) else far_plane

        # 2. Cost volume (needs scalar near/far for depth plane generation)
        cost_volume = self.cost_volume_constructor(
            features=features, # TF features only
            Ps=proj_matrices
        )

        features = features.reshape(B * K, self.cfg.feature_dim, H//4, W//4)    # [B*K, 128, H//4, W//4]
        features_cnn = features_cnn.reshape(B * K, self.cfg.feature_dim, H//4, W//4)  # [B*K, 128, H//4, W//4]

        upsampled_features_all = self.feature_upsampler(
            torch.cat([features, features_cnn], dim=1)
        ) # [B*K, C, H, W]

        # 3. Depth estimation (needs scalar near/far for inv_depth linspace)
        depth_maps, depth_conf = self.depth_estimator(
            cost_volume=cost_volume,
            images=context_images,
            features=upsampled_features_all,
        )
        depth_maps = torch.clamp(depth_maps, min=n, max=f)

        # 4. Gaussian/Convex splat head
        splats = self.splat_head(
            depth_map=depth_maps,
            depth_conf=depth_conf,
            images=context_images,
            features=upsampled_features_all.reshape(B, K, self.cfg.feature_dim, H, W).permute(0, 1, 3, 4, 2), # [B, K, H, W, C]
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
            global_step=global_step,
        )

        B_g, V_g = splats.means.shape[:2]
        rays = H * W
        srf = self.cfg.num_surfaces
        gpp = self.cfg.gaussians_per_pixel
        G = V_g * rays * srf * gpp

        # Flatten the (V, H*W, srf, gpp, ...) spatial dims into a single G dimension.
        splats.means      = splats.means.reshape(B_g, G, 3)
        splats.harmonics  = splats.harmonics.reshape(B_g, G, 3, -1)
        splats.opacities  = splats.opacities.reshape(B_g, G)

        if self.cfg.use_convex:
            K = self.cfg.nb_points
            splats.convex_points = splats.convex_points.reshape(B_g, G, K, 3)
            splats.delta         = splats.delta.reshape(B_g, G, 1)
            splats.sigma         = splats.sigma.reshape(B_g, G, 1)

            rendered = self.decoder(
                convex_splats=splats,
                extrinsics=target_extrinsics,
                intrinsics=target_intrinsics,
                near=near_plane,
                far=far_plane,
                image_shape=(H, W),
            )
        else:
            splats.covariances = splats.covariances.reshape(B_g, G, 3, 3)

            depth_mode = "depth" if render_depth else None
            rendered = self.decoder(
                gaussians=splats,
                extrinsics=target_extrinsics,
                intrinsics=target_intrinsics,
                near=near_plane,
                far=far_plane,
                image_shape=(H, W),
                depth_mode=depth_mode,
            )

        return {
            "rendered_images": rendered["color"],
            "rendered_depth": rendered["depth"],
            "depth_maps": depth_maps,
            "depth_conf": depth_conf,
            "splats": splats,
        }
