"""
Center crop + resize dataset images to a target size.
Datasets like MipNerf360 and Tanks & Temples may require this since
their image resolutions may not be multiplies of 16 (which is required by MVSplat).

The processing order is: center crop first (to maintain aspect ratio), then resize.
"""

import torch
import torchvision.transforms.functional as TF
from ..view_sampler.view_sampler import ViewSet

def update_intrinsics_for_resize(intrinsics: torch.Tensor, in_shape: tuple[int, int], out_shape: tuple[int, int]) -> torch.Tensor:
    H_in, W_in = in_shape
    H_out, W_out = out_shape
    K = intrinsics.clone()
    sx, sy = W_out / W_in, H_out / H_in
    K[:, 0, 0] *= sx
    K[:, 0, 2] *= sx
    K[:, 1, 1] *= sy
    K[:, 1, 2] *= sy
    return K

def update_intrinsics_for_crop(intrinsics: torch.Tensor, in_shape: tuple[int, int], out_shape: tuple[int, int]) -> torch.Tensor:
    H_in, W_in = in_shape
    H_out, W_out = out_shape
    K = intrinsics.clone()
    off_x = int(round((W_in - W_out) / 2.0))
    off_y = int(round((H_in - H_out) / 2.0))
    K[:, 0, 2] -= off_x
    K[:, 1, 2] -= off_y
    return K

def resize_images(
    images: torch.Tensor, # (B, 3, H, W)
    out_shape: tuple[int, int],
    intrinsics: torch.Tensor | None, # (B, 3, 3)
) -> tuple[torch.Tensor, torch.Tensor | None]:
    H_in, W_in = images.shape[-2:]
    H_out, W_out = out_shape
    new_images = TF.resize(images, out_shape, interpolation=TF.InterpolationMode.BILINEAR)
    if intrinsics is not None:
        K = update_intrinsics_for_resize(intrinsics, (H_in, W_in), (H_out, W_out))
    else:
        K = None
    return new_images, K

def center_crop_images(
    images: torch.Tensor, # (B, 3, H, W)
    out_shape: tuple[int, int],
    intrinsics: torch.Tensor | None, # (B, 3, 3)
) -> tuple[torch.Tensor, torch.Tensor | None]:
    H_in, W_in = images.shape[-2:]
    H_out, W_out = out_shape
    cropped = TF.center_crop(images, out_shape)
    if intrinsics is not None:
        K = update_intrinsics_for_crop(intrinsics, (H_in, W_in), (H_out, W_out))
    else:
        K = None
    return cropped, K

def resize_and_crop_images(
    images: torch.Tensor,         # (B, 3, H, W)
    out_shape: tuple[int, int],   # (H_out, W_out)
    intrinsics: torch.Tensor | None = None,     # (B, 3, 3)
    to_multiple_of_16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Center crop first, then resize images to target size, updating intrinsics."""
    H_in, W_in = images.shape[-2:]
    H_out, W_out = out_shape

    # Step 1: Center crop to a square (maintain aspect ratio)
    # Crop to the smaller dimension to preserve aspect ratio
    crop_size = min(H_in, W_in)
    cropped, K = center_crop_images(images, (crop_size, crop_size), intrinsics)
    
    # Step 2: Resize the cropped square image to target size
    if to_multiple_of_16:
        resize_shape = round_to_multiple_of_16(H_out, W_out)
    else:
        resize_shape = (H_out, W_out)
    
    resized, K = resize_images(cropped, resize_shape, K)
    return resized, K

def round_to_multiple_of_16(H: int, W: int) -> tuple[int, int]:
    """Get closest resolutions that are multiples of 16 to fit convolution requirements."""
    H_rounded = round(H / 16) * 16
    W_rounded = round(W / 16) * 16
    return H_rounded, W_rounded

def apply_crop_shim_to_views(views: ViewSet, shape: tuple[int, int], mulitple_of_16: bool = True) -> ViewSet:
    crop_images, crop_intrinsics = resize_and_crop_images(views.images, shape, views.intrinsics, mulitple_of_16)
    return ViewSet(extrinsics=views.extrinsics, intrinsics=crop_intrinsics, images=crop_images)