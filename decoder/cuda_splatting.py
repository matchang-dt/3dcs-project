import torch
from einops import einsum, rearrange, repeat
from math import sqrt
from typing import Literal
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.projection import to_homogeneous, get_fov, get_projection_matrix, depth_to_relative_disparity


def render_gaussians_cuda(
    extrinsics: torch.Tensor, # (B,V,4,4)
    intrinsics: torch.Tensor, # (B,V,3,3)
    near: torch.Tensor, # (B,V)
    far: torch.Tensor, # (B,V)
    image_shape: tuple[int, int], # (H,W)
    background_color: torch.Tensor, # (B,V,3)
    gaussian_means: torch.Tensor, # (B,V,gaussian,3)
    gaussian_covariances: torch.Tensor, # (B,V,gaussian,3,3)
    gaussian_sh_coefficients: torch.Tensor, # (B,V,gaussian,3,d_sh)
    gaussian_opacities: torch.Tensor, # (B,V,gaussian)
    use_sh: bool = True,
) -> torch.Tensor: # (B,V,3,H,W)
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # for numerical stability
    scale = 1 / near
    extrinsics = extrinsics.clone()
    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
    gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
    gaussian_means = gaussian_means * scale[:, None, None]
    near = near * scale
    far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = int(sqrt(n)) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    B = extrinsics.shape[0]
    H, W = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(B):
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i, ...],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

def render_depth_gaussians_cuda(
    extrinsics: torch.Tensor, # (B,V,4,4)
    intrinsics: torch.Tensor, # (B,V,3,3)
    near: torch.Tensor, # (B,V)
    far: torch.Tensor, # (B,V)
    image_shape: tuple[int, int], # (H,W)
    gaussian_means: torch.Tensor, # (B,V,gaussian,3)
    gaussian_covariances: torch.Tensor, # (B,V,gaussian,3,3)
    gaussian_opacities: torch.Tensor, # (B,V,gaussian)
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> torch.Tensor: # (B,V,H,W)
    camera_space_gaussians = einsum(
        extrinsics.inverse(), to_homogeneous(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # render using depth as color
    B = fake_color.shape[0]
    result = render_gaussians_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((B, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        use_sh=False,
    )
    return result.mean(dim=1)