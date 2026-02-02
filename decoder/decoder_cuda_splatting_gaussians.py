import torch
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Literal
from datasets.dataset import DatasetCfg
from gaussian_adapter.gaussians import Gaussians
from .decoder import Decoder
from .cuda_splatting import DepthRenderingMode, render_gaussians_cuda, render_depth_gaussians_cuda

"""
Actual gateway to render Gaussians with dcharatan's diffrast variant:
  https://github.com/dcharatan/diff-gaussian-rasterization-modified
"""

@dataclass
class DecoderGaussianSplattingCUDACfg:
    name: Literal["cuda_gaussian_splatting"]


class DecoderGaussianSplattingCUDA(Decoder[DecoderGaussianSplattingCUDACfg]):
    def __init__(self, cfg: DecoderGaussianSplattingCUDACfg, dataset_cfg: DatasetCfg):
        super().__init__(cfg, dataset_cfg)
        self.background_color = [0,0,0] if getattr(dataset_cfg, "background_color", None) is None \
                                else dataset_cfg.background_color
        self.background_color = torch.tensor(self.background_color, dtype=torch.float32)

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: torch.Tensor, # (B,V,4,4)
        intrinsics: torch.Tensor, # (B,V,3,3)
        near: torch.Tensor, # (B,V)
        far: torch.Tensor, # (B,V)
        image_shape: tuple[int, int], # (H,W)
        depth_mode: DepthRenderingMode | None = None,
    ):
        B, V = extrinsics.shape[:2]
        color = render_gaussians_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color.to(extrinsics.device), "c -> (b v) c", b=B, v=V),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=V),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=V),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=V),
            repeat(gaussians.opacities, "b g -> (b v) g", v=V),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=B, v=V)
        depth = None
        if depth_mode is not None:
            depth = self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            )

        return {
            "color": color,
            "depth": depth,
        }

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: torch.Tensor, # (B,V,4,4)
        intrinsics: torch.Tensor, # (B,V,3,3)
        near: torch.Tensor, # (B,V)
        far: torch.Tensor, # (B,V)
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> torch.Tensor: # (B,V,H,W)
        B, V = extrinsics.shape[:2]
        result = render_depth_gaussians_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=V),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=V),
            repeat(gaussians.opacities, "b g -> (b v) g", v=V),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=B, v=V)