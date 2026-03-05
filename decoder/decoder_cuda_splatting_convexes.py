import torch
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Literal

from datasets.dataset import DatasetCfg
from convex_adapter.convex_splats import ConvexSplats
from .decoder import Decoder
from .cuda_convex_splatting import render_convexes_cuda


@dataclass
class DecoderConvexSplattingCUDACfg:
    name: Literal["cuda_convex_splatting"]
    sh_degree: int = 4


class DecoderConvexSplattingCUDA(Decoder[DecoderConvexSplattingCUDACfg]):
    """
    Gateway to the differentiable convex-splat rasterizer.

    Mirrors DecoderGaussianSplattingCUDA exactly in interface:
      - receives a ConvexSplats object (analogous to Gaussians)
      - broadcasts the splats to all target views
      - returns {"color": (B, V, 3, H, W), "depth": None}

    The underlying render_convexes_cuda call uses the same camera math and the
    same 1/near stability scaling as the Gaussian path.
    """

    def __init__(self, cfg: DecoderConvexSplattingCUDACfg, dataset_cfg: DatasetCfg):
        super().__init__(cfg, dataset_cfg)
        self.sh_degree = cfg.sh_degree
        bg = (
            [0, 0, 0]
            if getattr(dataset_cfg, "background_color", None) is None
            else dataset_cfg.background_color
        )
        self.background_color = torch.tensor(bg, dtype=torch.float32)

    def forward(
        self,
        convex_splats: ConvexSplats,
        extrinsics: torch.Tensor,   # (B, V, 4, 4)
        intrinsics: torch.Tensor,   # (B, V, 3, 3)
        near: torch.Tensor,         # (B, V)
        far: torch.Tensor,          # (B, V)
        image_shape: tuple[int, int],
    ) -> dict:
        B, V = extrinsics.shape[:2]
        bg_color = self.background_color.to(device=extrinsics.device, dtype=torch.float32)

        color = render_convexes_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j").contiguous(),
            rearrange(intrinsics, "b v i j -> (b v) i j").contiguous(),
            rearrange(near, "b v -> (b v)").contiguous(),
            rearrange(far,  "b v -> (b v)").contiguous(),
            image_shape,
            repeat(bg_color, "c -> (b v) c", b=B, v=V).contiguous(),
            repeat(convex_splats.convex_points, "b n k xyz -> (b v) n k xyz", v=V).contiguous(),
            repeat(convex_splats.delta,         "b n one -> (b v) n one",     v=V).contiguous(),
            repeat(convex_splats.sigma,         "b n one -> (b v) n one",     v=V).contiguous(),
            repeat(convex_splats.opacities,     "b n -> (b v) n",             v=V).contiguous(),
            repeat(convex_splats.harmonics,     "b n c d -> (b v) n c d",     v=V).contiguous(),
            sh_degree=self.sh_degree,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=B, v=V)
        return {"color": color, "depth": None}
