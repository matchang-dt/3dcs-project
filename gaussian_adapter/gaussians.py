import torch
from dataclasses import dataclass
from einops import rearrange
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from math import sqrt

@dataclass
class Gaussians:
    means: torch.Tensor # (B,G,3)
    scales: torch.Tensor # (B,G,3)
    covariances: torch.Tensor # (B,G,3,3)
    rotations: torch.Tensor # (B,G,4)
    opacities: torch.Tensor # (B,G)
    harmonics: torch.Tensor # (B,G,3,deg(sh)+1)


# from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def construct_covariance_matrix(
    scale: torch.Tensor,
    quaternion: torch.Tensor
) -> torch.Tensor:
    # scale (..., 3) -> (..., 3, 3) diagonal matrix
    scale_diag = scale.unsqueeze(-1) * torch.eye(3, device=scale.device, dtype=scale.dtype)
    rotation = quaternion_to_matrix(quaternion)
    return rotation @ scale_diag @ scale_diag.transpose(-1, -2) @ rotation.transpose(-1, -2)


def rotate_sh(sh: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    # Wigner-D, basically rotations for spherical harmonics
    # sh: (B, G, 3, sh_dim), sh_rotations: (B, 2*deg+1, 2*deg+1)
    n = sh.shape[-1]

    alpha, beta, gamma = matrix_to_angles(rotations)
    rotated_sh_list = []
    for deg in range(int(sqrt(n))):
        sh_rotations = wigner_D(deg, alpha, beta, gamma).to(sh.device).to(sh.dtype)
        sh_slice = sh[..., deg ** 2 : (deg + 1) ** 2]
        # (B, i, j) @ (B, G, 3, j) -> (B, G, 3, i)
        sh_rotated = torch.einsum("...ij,...cj->...ci", sh_rotations, sh_slice)
        rotated_sh_list.append(sh_rotated)
    return torch.cat(rotated_sh_list, dim=-1)