import torch
from jaxtyping import Float
from torch import Tensor
from einops import einsum

def to_homogeneous(points: torch.Tensor, at_infinity: bool = False) -> torch.Tensor:
    # to xyzw homogeneous coordinates
    concat_array = torch.ones_like(points[..., :1]) # w=1
    if at_infinity: # used for vectors
        concat_array = torch.zeros_like(concat_array) # w=0
    return torch.cat([points, concat_array], dim=-1)

def get_fov(intrinsics: torch.Tensor) -> torch.Tensor:
    """Get FoV for x and y axes for rendering."""
    # intrinsics (B,3,3)
    intrinsics_inv = torch.linalg.inv(intrinsics)

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)

def depth_to_relative_disparity(
    depth: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Convert depth to relative disparity, where 0 is near and 1 is far"""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    disp = 1 / (depth + eps)
    return 1 - (disp - disp_far) / (disp_near - disp_far + eps)

def get_projection_matrix(
    near: torch.Tensor,
    far: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
) -> torch.Tensor:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    B = near.shape[0]
    result = torch.zeros((B, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result
