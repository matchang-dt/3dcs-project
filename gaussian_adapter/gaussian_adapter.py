from dataclasses import dataclass
from .gaussians import Gaussians, construct_covariance_matrix, rotate_sh
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import einsum
from typing import Tuple


@dataclass
class GaussianAdapterCfg:
    sh_degree: int = 4
    scale_min: float = 0.01
    scale_max: float = 100
    gaussian_scale_pct: float = 0.1  # heuristic used in orig. MVSplat
    gaussians_per_pixel: int = 1
    num_surfaces: int = 1

def to_homogeneous(points: torch.Tensor, at_infinity: bool = False) -> torch.Tensor:
    # to xyzw homogeneous coordinates
    concat_array = torch.ones_like(points[..., :1]) # w=1
    if at_infinity: # used for vectors
        concat_array = torch.zeros_like(concat_array) # w=0
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
    - pixel_centers: (B, V, R, srf, gpp, 2)
    - extrinsics: (B, V, 1, 1, 1, 4, 4)
    - intrinsics: (B, V, 1, 1, 1, 3, 3)

    Output:
    - ray_o: (B, V, R, srf, gpp, 3)
    - ray_d: (B, V, R, srf, gpp, 3)
    """
    # 'ray trace': (B, 3, 3) @ (B, G, 3) -> (B, G, 3)
    hom_pixel_centers = to_homogeneous(pixel_centers)
    ray_d = einsum(
        torch.linalg.inv(intrinsics),
        hom_pixel_centers,
        "... i j, ... j -> ... i"
    )
    ray_d = F.normalize(ray_d, dim=-1)
    # (B, 4, 4) @ (B, G, 4) -> (B, G, 4)
    ray_d = einsum(
        extrinsics,
        to_homogeneous(ray_d, at_infinity=True),
        "... i j, ... j -> ... i"
    )
    ray_d = ray_d[..., :-1] # dehomogenize

    # ray origins: (B, V, 1, 1, 1, 3) -> (B, V, R, srf, gpp, 3)
    ray_o = extrinsics[..., :-1, -1].expand_as(ray_d)
    return ray_o, ray_d

class GaussianAdapter(torch.nn.Module):
    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg
        self.sh_dim = (self.cfg.sh_degree + 1) ** 2

    def _prepare_broadcast_amenable_shapes(
        self,
        pre_gaussians: torch.Tensor, # orig. (B,V,rays=H*W,1,84=2+(3+4+3*sh_dim=25))
        pixel_centers: torch.Tensor, # (B,V,rays,1,2)
        extrinsics: torch.Tensor, # orig. (B,V,4,4) -> (B,V,1,1,1,4,4)
        intrinsics: torch.Tensor, # orig. (B,V,3,3) -> (B,V,1,1,1,3,3)
        opacities: torch.Tensor, # orig. (B,V,rays,1,1), no change
        depths: torch.Tensor, # orig. (B,V,rays,1,1), no change
    ): # returns all the arguments, reshaped for broadcast operations
        srf = self.cfg.num_surfaces
        gpp = self.cfg.gaussians_per_pixel
        pre_gaussians = pre_gaussians[..., None, :].expand(-1, -1, -1, srf, gpp, -1)
        pixel_centers = pixel_centers[..., None, :].expand(-1, -1, -1, srf, gpp, -1)
        extrinsics = extrinsics[:, :, None, None, None, :, :].expand(-1, -1, -1, srf, gpp, -1, -1)
        intrinsics = intrinsics[:, :, None, None, None, :, :].expand(-1, -1, -1, srf, gpp, -1, -1)
        if opacities.ndim == 3: # if (B,V,rays), provide the last two dims
            opacities = opacities[..., None, None].expand(-1,-1,-1,srf,gpp)
        if depths.ndim == 3: # if (B,V,r), provide last two dims
            depths = depths[..., None, None].expand(-1,-1,-1,srf,gpp)
        return pre_gaussians, pixel_centers, extrinsics, intrinsics, opacities, depths

    def forward(
        self,
        pre_gaussians: torch.Tensor, # orig. (B,V,rays=H*W,srf=1,84=2+(3+4+3*sh_dim=25))
        pixel_centers: torch.Tensor, # (B,V,rays,1,2)
        extrinsics: torch.Tensor, # orig. (B,V,4,4) -> (B,V,1,1,1,4,4)
        intrinsics: torch.Tensor, # orig. (B,V,3,3) -> (B,V,1,1,1,3,3)
        opacities: torch.Tensor, # orig. (B,V,rays,srf=1,gpp=1)
        depths: torch.Tensor, # orig. (B,V,rays,1,1)
        img_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Input (pre-set to broadcast-compatible shapes):
        - pre_gaussians: (batch_dim(B), views(V), num_rays(R=H*W), num_surfaces(srf), gaussians_per_pixel(gpp), c)
        - pixel_centers: (B, V, R, srf, gpp, 2)
        - extrinsics: (B, V, 1, 1, 1, 4, 4)
        - intrinsics: (B, V, 1, 1, 1, 3, 3)
        - opacities: (B, V, R, srf, gpp, spp)
        - depths: (B, V, R, srf, gpp, spp)
        - img_shape: (H, W)

        Output: Gaussians(means, covariances, rotations, opacities, harmonics)
        """
        H, W = img_shape
        B, V = pre_gaussians.shape[:2]
        gpp = self.cfg.gaussians_per_pixel
        srf = self.cfg.num_surfaces

        pre_gaussians, pixel_centers, extrinsics, intrinsics, opacities, depths = self._prepare_broadcast_amenable_shapes(
            pre_gaussians, pixel_centers, extrinsics, intrinsics, opacities, depths
        )

        # split logits into separate parameters
        # first 2 vals of pre_gaussians are xy offsets (so last dim should be 82 by default)
        scales, quaternions, sh = torch.split(pre_gaussians, (3,4,3 * self.sh_dim), dim=-1)

        # bound and squeeze depths into a certain scene range
        # depths used for perspective projection (as a pixel covers more space further away)
        scene_range = self.cfg.scale_max - self.cfg.scale_min
        scales = (self.cfg.scale_min + scene_range * torch.sigmoid(scales)) * depths.unsqueeze(-1)
        pixel_size = 1. / torch.tensor([W, H], device=depths.device)
        scale_adjustment = self.cfg.gaussian_scale_pct * torch.einsum(
            "...ij,j->...",
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
        )
        scales = scales * scale_adjustment.unsqueeze(-1)
        # scales: (B,V,R,num_surfaces=1,gaussians_per_pixel=1,3)

        # normalize to get actual quaternion from "quaternion logits"
        quaternions = F.normalize(quaternions, dim=-1)
        # quaternions: (B,V,R,surfaces,gpp,4)
        
        # covariance to world space
        R_w2c = extrinsics[..., :3, :3]
        R_c2w = R_w2c.transpose(-1, -2)
        covariances = construct_covariance_matrix(scales, quaternions)
        covariances = R_c2w @ covariances @ R_c2w.transpose(-1, -2)
        # covariances: (B,V,R,srf,gpp,3,3)

        # gaussian means from 'ray tracing'/unprojecting based on depth
        ray_o, ray_d = get_camera_rays_world(pixel_centers, extrinsics, intrinsics)
        means = ray_o + ray_d * depths.unsqueeze(-1)
        # ray_o, ray_d: (B,V,R,srf,gpp,3)
        # means: (B,V,R,srf,gpp,3)

        # spherical harmonics need to be expanded and set to world space
        sh = rearrange(sh, '... (x sh_dim) -> ... x sh_dim', x=3)
        sh = sh.expand(B, V, H*W, srf, gpp, 3, self.sh_dim)
        sh = rotate_sh(sh, R_c2w)
        # sh: (B,V,R,srf,gpp,3,sh_dim)

        return Gaussians(
            means=means,
            scales=scales,
            covariances=covariances,
            rotations=quaternions,
            opacities=opacities,
            harmonics=sh,
        )
