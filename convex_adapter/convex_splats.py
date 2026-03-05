import torch
from dataclasses import dataclass


@dataclass
class ConvexSplats:
    """
    Batched convex splat representation, analogous to Gaussians.

    Each splat is defined by K 3D vertices (whose projected convex hull
    defines the 2-D shape) plus two sharpness scalars delta and sigma that
    control the smooth inside/outside function in the diff-convex-rasterizer.

    All tensors are in world space and carry float32 values as required by
    the CUDA rasterization kernel.
    """

    means: torch.Tensor          # (B, N, 3)      centroid of the splat (world space)
    convex_points: torch.Tensor  # (B, N, K, 3)   K vertices per splat (world space; relative to mean/centroid)
    delta: torch.Tensor          # (B, N, 1)      sharpness  (after exp activation)
    sigma: torch.Tensor          # (B, N, 1)      smoothness   (after exp activation)
    opacities: torch.Tensor      # (B, N)         base opacity in [0, 1]
    harmonics: torch.Tensor      # (B, N, 3, sh_dim)  spherical-harmonic colour coefficients
