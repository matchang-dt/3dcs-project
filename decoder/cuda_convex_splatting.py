import torch
from einops import rearrange

from diff_convex_rasterization import ConvexRasterizationSettings, ConvexRasterizer
from utils.projection import get_fov, get_projection_matrix


def render_convexes_cuda(
    extrinsics: torch.Tensor,       # (BV, 4, 4)  world-to-camera (w2c)
    intrinsics: torch.Tensor,       # (BV, 3, 3)
    near: torch.Tensor,             # (BV,)
    far: torch.Tensor,              # (BV,)
    image_shape: tuple[int, int],   # (H, W)
    background_color: torch.Tensor, # (BV, 3)
    convex_points: torch.Tensor,    # (BV, N, K, 3)  K vertices per splat
    delta: torch.Tensor,            # (BV, N, 1)  interior sharpness (post exp activation)
    sigma: torch.Tensor,            # (BV, N, 1)  boundary fall-off  (post exp activation)
    opacities: torch.Tensor,        # (BV, N)
    sh_coefficients: torch.Tensor,  # (BV, N, 3, sh_dim)
    sh_degree: int = 4,
) -> torch.Tensor:                  # (BV, 3, H, W)
    """
    Differentiable convex-splat rasterizer for a flat batch of BV views.

    Mirrors render_gaussians_cuda in cuda_splatting.py:
      - scales all 3-D geometry by 1/near for numerical stability
      - builds ConvexRasterizationSettings from the same view / projection matrices
      - loops over BV views and calls ConvexRasterizer for each one

    All splats are assumed to have exactly K vertices (constant topology), which
    lets us pre-compute num_points_per_convex and cumsum_of_points_per_convex
    once and reuse them across the whole batch.
    """
    original_dtype = convex_points.dtype

    # force fp32
    convex_points   = convex_points.float()
    delta           = delta.float()
    sigma           = sigma.float()
    opacities       = opacities.float()
    sh_coefficients = sh_coefficients.float()
    extrinsics      = extrinsics.float()
    intrinsics      = intrinsics.float()
    near            = near.float()
    far             = far.float()
    background_color = background_color.float()

    # scale for numerical stability
    scale = 1.0 / near                              # (BV,)
    extrinsics = extrinsics.clone()
    extrinsics = torch.linalg.inv(extrinsics)       # w2c → c2w
    extrinsics[..., :3, 3] *= scale[:, None]
    convex_points = convex_points * scale[:, None, None, None]
    near = near * scale
    far  = far  * scale

    # build view / projection matrices
    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix       = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection   = view_matrix @ projection_matrix

    # convex rasterizer expects SH of (N, sh_dim, 3)
    # our harmonics are stored as (BV, N, 3, sh_dim) -> transpose last two dims
    shs = rearrange(sh_coefficients, "b n xyz sh -> b n sh xyz").contiguous()

    BV = extrinsics.shape[0]
    H, W = image_shape
    N = convex_points.shape[1]
    K = convex_points.shape[2]

    # constant K vertices per splat
    num_pts    = torch.full((N,), K, dtype=torch.int32, device=convex_points.device)
    cumsum_pts = torch.arange(0, N * K, K, dtype=torch.int32, device=convex_points.device)

    all_images = []
    for i in range(BV):
        pts_flat = convex_points[i].reshape(-1).contiguous() # (NK*3,)

        # Clone delta / sigma and mark for grad retention (mirrors convex_renderer)
        delta_i = delta[i].clone().requires_grad_(True)
        sigma_i = sigma[i].clone().requires_grad_(True)
        try:
            delta_i.retain_grad()
            sigma_i.retain_grad()
        except Exception:
            pass

        # Placeholder for 2-D screen-space positions (grad hook, same as Gaussian means2D)
        means2D = torch.zeros(
            (N, 3), device=pts_flat.device, dtype=torch.float32, requires_grad=True
        )
        try:
            means2D.retain_grad()
        except Exception:
            pass

        # Pre-allocated output buffers written by the CUDA kernel in-place
        scaling        = torch.zeros(N, device=pts_flat.device, dtype=torch.float32)
        density_factor = torch.zeros(N, device=pts_flat.device, dtype=torch.float32)

        settings = ConvexRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=sh_degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = ConvexRasterizer(raster_settings=settings)

        rendered_image, _, _, _, _ = rasterizer(
            convex_points=pts_flat,
            delta=delta_i,
            sigma=sigma_i,
            num_points_per_convex=num_pts,
            cumsum_of_points_per_convex=cumsum_pts,
            number_of_points=N,
            opacities=opacities[i, :, None],
            means2D=means2D,
            scaling=scaling,
            density_factor=density_factor,
            shs=shs[i],
            colors_precomp=None,
        )
        all_images.append(rendered_image)

    result = torch.stack(all_images)  # (BV, 3, H, W)
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    return result
