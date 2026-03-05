from dataclasses import dataclass
import torch
from einops import rearrange

from gaussian_adapter.gaussians import rotate_sh
from utils.projection import get_camera_rays_world
from .convex_splats import ConvexSplats


@dataclass
class ConvexAdapterCfg:
    sh_degree: int = 4
    nb_points: int = 6           # K: number of vertices per splat (max 8 per CUDA config)
    scale_min: float = 0.01      # minimum vertex-offset scale (sigmoid center)
    scale_max: float = 15.0      # maximum vertex-offset scale (sigmoid range)
    splat_scale_pct: float = 0.1 # world-units-per-pixel heuristic (like gaussian_scale_pct)
    splats_per_pixel: int = 1
    num_surfaces: int = 1


class ConvexAdapter(torch.nn.Module):
    """
    Converts per-pixel NN predictions into batched world-space ConvexSplats.

    Analogous to GaussianAdapter.  Instead of predicting (scale, quaternion)
    to form a covariance ellipsoid, we predict:
      - K 3-D vertex offsets in camera space  →  rotated to world space and
        added to the ray-cast mean to give the K vertices of the splat.
      - log_delta, log_sigma  →  exp-activated sharpness parameters for the
        differentiable convex rasterizer.
      - spherical-harmonic coefficients (same as Gaussian path).
    """

    def __init__(self, cfg: ConvexAdapterCfg):
        super().__init__()
        self.cfg = cfg
        self.sh_dim = (cfg.sh_degree + 1) ** 2

    def _broadcast_inputs(
        self,
        pre_splats,
        pixel_centers,
        extrinsics,
        intrinsics,
        opacities,
        depths,
    ):
        srf = self.cfg.num_surfaces
        gpp = self.cfg.splats_per_pixel
        pre_splats    = pre_splats[..., None, :].expand(-1, -1, -1, srf, gpp, -1)
        pixel_centers = pixel_centers[..., None, :].expand(-1, -1, -1, srf, gpp, -1)
        extrinsics    = extrinsics[:, :, None, None, None, :, :].expand(-1, -1, -1, srf, gpp, -1, -1)
        intrinsics    = intrinsics[:, :, None, None, None, :, :].expand(-1, -1, -1, srf, gpp, -1, -1)
        if opacities.ndim == 3:
            opacities = opacities[..., None, None].expand(-1, -1, -1, srf, gpp)
        if depths.ndim == 3:
            depths = depths[..., None, None].expand(-1, -1, -1, srf, gpp)
        return pre_splats, pixel_centers, extrinsics, intrinsics, opacities, depths

    def forward(
        self,
        pre_splats: torch.Tensor,    # (B, V, R, srf, K*3+2+3*sh_dim) raw NN predictions
        pixel_centers: torch.Tensor, # (B, V, R, srf, 2)
        extrinsics: torch.Tensor,    # (B, V, 4, 4)  world-to-camera
        intrinsics: torch.Tensor,    # (B, V, 3, 3)
        opacities: torch.Tensor,     # (B, V, R)
        depths: torch.Tensor,        # (B, V, R)
        img_shape: tuple[int, int],
    ) -> ConvexSplats:
        H, W = img_shape
        B, V = pre_splats.shape[:2]
        K = self.cfg.nb_points

        pre_splats, pixel_centers, extrinsics, intrinsics, opacities, depths = (
            self._broadcast_inputs(
                pre_splats, pixel_centers, extrinsics, intrinsics, opacities, depths
            )
        )
        # All tensors are now (B, V, R, srf, gpp, ...)

        # pre_splats: [vertex_offsets(K*3) | smoothness(1) | sharpness(1) | sh(3*sh_dim)]
        vertex_offsets_raw, log_delta, log_sigma, sh_raw = torch.split(
            pre_splats, (K * 3, 1, 1, 3 * self.sh_dim), dim=-1
        )

        # Reshape vertex offsets: (..., K*3) -> (..., K, 3)
        vertex_offsets_raw = rearrange(
            vertex_offsets_raw, "... (k xyz) -> ... k xyz", k=K
        )

        # scale vertex offsets by depth * pixel footprint
        pixel_size = 1.0 / torch.tensor(
            [W, H], device=depths.device, dtype=depths.dtype
        )
        # pixel_scale: (...,) world-space size of one pixel at unit depth
        pixel_scale = self.cfg.splat_scale_pct * torch.einsum(
            "...ij,j->...",
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
        )

        # offset in camera space: centred (sigmoid-0.5), scaled by depth and pixel footprint
        scene_range = self.cfg.scale_max - self.cfg.scale_min
        vertex_offsets_cam = (
            scene_range
            * (torch.sigmoid(vertex_offsets_raw) - 0.5)
            * depths[..., None, None]
            * pixel_scale[..., None, None]
        )
        # vertex_offsets_cam: (..., K, 3) in camera space

        # ray means (would-be centers for gaussians)
        ray_o, ray_d = get_camera_rays_world(
            pixel_centers, extrinsics, intrinsics, img_shape
        )
        means = ray_o + ray_d * depths[..., None]
        # means: (..., 3)

        # rotate vertex offsets to world space and displace from mean
        R_w2c = extrinsics[..., :3, :3]
        R_c2w = R_w2c.transpose(-1, -2)
        vertex_offsets_world = torch.einsum(
            "...ij,...kj->...ki", R_c2w, vertex_offsets_cam
        )
        convex_points = means[..., None, :] + vertex_offsets_world
        # convex_points: (..., K, 3)

        # exp(sharpness & smoothness) to enforce positive definite
        delta = torch.exp(log_delta.clamp(-10.0, 10.0))  # (..., 1)
        sigma = torch.exp(log_sigma.clamp(-10.0, 10.0))  # (..., 1)

        # reshape and rotate spherical harmonics to world space
        sh = rearrange(sh_raw, "... (x sh_dim) -> ... x sh_dim", x=3)
        sh = sh.expand(
            B, V, H * W,
            self.cfg.num_surfaces, self.cfg.splats_per_pixel,
            3, self.sh_dim,
        )
        sh = rotate_sh(sh, R_c2w)
        # sh: (..., 3, sh_dim)

        return ConvexSplats(
            means=means.float(),
            convex_points=convex_points.float(),
            delta=delta.float(),
            sigma=sigma.float(),
            opacities=opacities.float(),
            harmonics=sh.float(),
        )
