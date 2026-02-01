from dataclasses import dataclass
from gaussians import Gaussians, construct_covariance_matrix, rotate_sh
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import einsum
from typing import Tuple


@dataclass
class GaussianAdapterCfg:
    sh_degree: int = 3
    scale_min: float = 0.01
    scale_max: float = 100
    gaussian_scale_pct: float = 0.1  # heuristic used in orig. MVSplat

def to_homogeneous(points: torch.Tensor, at_infinity: bool = False) -> torch.Tensor:
    concat_array = torch.ones_like(points[..., :1])
    if at_infinity:
        concat_array = torch.zeros_like(concat_array)
    return torch.cat([points, concat_array], dim=-1)

# maybe move this to a separate file
def get_camera_rays_world(
    pixel_centers: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get rays extending from camera center to pixel centers, in *world space*.

    Input:
    - pixel_centers: (B, G, 2)
    - extrinsics: (B, 4, 4)
    - intrinsics: (B, 3, 3)

    Output:
    - ray_o: (B, G, 3)
    - ray_d: (B, G, 3)
    """
    # 'ray trace'
    hom_pixel_centers = to_homogeneous(pixel_centers)
    # ray directions
    ray_d = einsum(
        torch.linalg.inv(intrinsics),
        hom_pixel_centers,
        "... i j, ... j -> ... i"
    )
    ray_d = F.normalize(ray_d, dim=-1, keepdim=True)
    ray_d = einsum(
        extrinsics,
        to_homogeneous(ray_d, at_infinity=True),
        "... i j, ... j -> ... i"
    )
    ray_d = ray_d[..., :-1]

    # ray origins
    ray_o = extrinsics[..., :-1, -1].expand(ray_d.shape) # 't' component
    return ray_o, ray_d

class GaussianAdapter(torch.nn.Module):
    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg
        self.sh_dim = (self.cfg.sh_degree + 1) ** 2

    def forward(
        self,
        pre_gaussians: torch.Tensor,
        pixel_centers: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        img_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Input:
        - pre_gaussians: post-UNet gaussian head 'logits' (B, G, 3)
        - pixel_centers: centers for each pixel in B images (B, G, 3)
        - extrinsics: (B, 4, 4)
        - intrinsics: (B, 3, 3)
        - opacities: (B, G)
        - depths: (B, G)
        - img_shape: (H, W)

        Output: Gaussians(means, covariances, rotations, opacities, harmonics, colors)
        """
        H, W = img_shape
        B, G = depths.shape
        
        # split logits into separate parameters
        scales, quaternions, sh = torch.split(pre_gaussians, (3,4,3 * self.sh_dim), dim=2)

        # bound and squeeze depths into a certain scene range
        # depths used for perspective projection (as a pixel covers more space further away)
        scene_range = self.cfg.scale_max - self.cfg.scale_min
        scales = (self.cfg.scale_min + scene_range * torch.sigmoid(scales)) * depths.unsqueeze(-1)
        pixel_size = 1. / torch.tensor([W, H], device=depths.device)
        scale_adjustment = self.cfg.gaussian_scale_pct * einsum(
            intrinsics[..., :2, :2].inverse(), # focal lengths
            pixel_size,
            "...ij,j->..."
        )
        scales = scales * scale_adjustment.unsqueeze(-1)

        # normalize to get actual quaternion from "quaternion logits"
        quaternions = F.normalize(quaternions, dim=-1)

        # covariance to world space
        c2w = extrinsics[..., :3, :3]
        covariances = construct_covariance_matrix(scales, quaternions)
        covariances = c2w @ covariances @ c2w.T

        # gaussian means from 'ray tracing'/unprojecting based on depth
        ray_o, ray_d = get_camera_rays_world(pixel_centers, extrinsics, intrinsics)
        means = ray_o + ray_d * depths.unsqueeze(-1)

        # spherical harmonics need to be expanded and set to world space
        sh = rearrange(sh, '... (x sh_dim) -> ... x sh_dim', sh_dim=self.sh_dim)
        sh = sh.expand(B, G, -1, -1)
        sh = rotate_sh(sh, c2w)

        return Gaussians(
            means=means,
            scales=scales,
            covariances=covariances,
            rotations=quaternions,
            opacities=opacities,
            harmonics=sh,
        )
