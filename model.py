import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from extractor.extractor import Extractor
from cost_volume_constructor.constructor import CostVolumeConstructor
from depth_estimator.estimator import DepthEstimator
from gaussian_adapter.gaussian_head import GaussianHead, GaussianHeadConfig
from gaussian_adapter.gaussian_adapter import GaussianAdapterCfg
from decoder.decoder_cuda_splatting_gaussians import (
    DecoderGaussianSplattingCUDA,
    DecoderGaussianSplattingCUDACfg,
)
from datasets.dataset import DatasetCfg
from utils.projection import make_proj_matrix


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

    # Gaussian head params
    gaussian_head_channels: int = 132  # 128 (features) + 4 (RGB + depth)
    opacity_start: float = 0.5
    opacity_end: float = 2.0
    opacity_warmup: float = 10000

    # Gaussian adapter params
    sh_degree: int = 4
    scale_min: float = 0.01
    scale_max: float = 100.0
    gaussian_scale_pct: float = 0.1
    gaussians_per_pixel: int = 1
    num_surfaces: int = 1

    # Decoder params
    decoder_cfg: Optional[DecoderGaussianSplattingCUDACfg] = None
    dataset_cfg: Optional[DatasetCfg] = None

class MVSplat(nn.Module):
    """
    MVSplat: Multi-View Splat model for novel view synthesis.

    Pipeline:
    1. Extractor: Extract features from input images
    2. Cost Volume Constructor: Build cost volume from features
    3. Depth Estimator: Estimate depth maps from cost volume
    4. Gaussian Head: Predict gaussian parameters and convert to 3D gaussians
    5. Decoder: Render gaussian splats to images
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

        self.depth_estimator = DepthEstimator(
            near=cfg.near,
            far=cfg.far,
            channels=cfg.feature_dim,
            feat_map_size=cfg.image_size,
            dtype=cfg.pipeline_dtype,
        )

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

        self.gaussian_head = GaussianHead(gaussian_head_cfg)

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

        near_plane = batch.get("near_plane", 0.1)
        far_plane = batch.get("far_plane", 100.0)

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
        features = self.extractor(context_images)
        features = features.to(self.cfg.pipeline_dtype)

        with torch.autocast(device_type='cuda', enabled=False):
            proj_matrices = make_proj_matrix(context_extrinsics, context_intrinsics)
        proj_matrices = proj_matrices.to(self.cfg.pipeline_dtype)

        # 2. Cost volume
        cost_volume = self.cost_volume_constructor(
            features=features,
            Ps=proj_matrices,
        )

        # 3. Depth estimation
        depth_maps, depth_conf = self.depth_estimator(
            cost_volume=cost_volume,
            images=context_images,
            features=features,
        )
        n = near_plane.min() if isinstance(near_plane, torch.Tensor) else near_plane
        f = far_plane.max() if isinstance(far_plane, torch.Tensor) else far_plane
        depth_maps = torch.clamp(depth_maps, min=n+1e-4, max=f-1e-4)

        # 4. Gaussians (head calls adapter internally)
        gaussians = self.gaussian_head(
            depth_map=depth_maps,
            depth_conf=depth_conf,
            images=context_images,
            features=features,
            extrinsics=context_extrinsics,
            intrinsics=context_intrinsics,
            global_step=global_step,
        )

        B_g, V_g = gaussians.means.shape[:2]
        rays = H * W
        srf = self.cfg.num_surfaces
        gpp = self.cfg.gaussians_per_pixel
        G = V_g * rays * srf * gpp

        gaussians.means = gaussians.means.reshape(B_g, G, 3)
        gaussians.covariances = gaussians.covariances.reshape(B_g, G, 3, 3)
        gaussians.harmonics = gaussians.harmonics.reshape(B_g, G, 3, -1)
        gaussians.opacities = gaussians.opacities.reshape(B_g, G)

        depth_mode = "depth" if render_depth else None
        rendered = self.decoder(
            gaussians=gaussians,
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
            "gaussians": gaussians,
        }
