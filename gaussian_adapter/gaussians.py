import torch
from dataclasses import dataclass
from einops import rearrange
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from math import sqrt

@dataclass
class Gaussians:
    means: torch.Tensor # (G,3)
    scales: torch.Tensor # (G,3)
    covariances: torch.Tensor # (G,3,3)
    rotations: torch.Tensor # (G,4)
    opacities: torch.Tensor # (G,)
    harmonics: torch.Tensor # (G,3,deg(sh)+1)


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
    scale = scale * torch.eye(scale).expand(scale.shape[:-2], -1, -1)
    rotation = quaternion_to_matrix(quaternion)
    return rotation @ scale @ scale.T @ rotation.T


def rotate_sh(sh: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    # Wigner-D, basically rotations for spherical harmonics
    n = sh.shape[-1]

    alpha, beta, gamma = matrix_to_angles(rotations)
    rotated_sh_list = []
    for deg in range(int(sqrt(n))):
        sh_rotations = wigner_D(deg, alpha, beta, gamma).to(sh.device).to(sh.dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh[..., deg ** 2 : (deg + 1) ** 2],
            "...ij, ...j -> ...i"
        )
        rotated_sh_list.append(sh_rotated)
    return torch.cat(rotated_sh_list, dim=-1)