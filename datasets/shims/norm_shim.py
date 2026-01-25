"""
Shim that normalizes images and intrinsics.
"""

import torch
from typing import Literal


def already_normalized(tensor: torch.Tensor, type: Literal["images", "intrinsics"]) -> bool:
    """
    Check if image/intrinsic tensor is already normalized.
    Should be of sizes [B, 3, 3] for intrinsics, [B, 3, H, W] for images.
    NOTE: if image is from [0,1] then this will accept it (will not transform image.)
    If it is [-1,1] it will also accept. Ensure it's the range you need it to be in.
    """
    if type == "images":
        return tensor.max() <= 1.0 and tensor.min() >= -1.0
    elif type == "intrinsics":
        cx, cy = tensor[:, 0, 2], tensor[:, 1, 2]
        return cx.max() <= 1.0 and cx.min() >= -1.0 and cy.max() <= 1.0 and cy.min() >= -1.0
    else:
        raise ValueError(f"Invalid type: {type}")

def normalize_images(images: torch.Tensor, interval: tuple[float, float] = (0, 1)) -> torch.Tensor:
    """
    Normalizes images to interval provided.
    Assumes images are in [0, 255] or similar range and normalizes to [interval[0], interval[1]].
    """
    # check first
    if already_normalized(images, "images"):
        return images
    
    # Compute actual min/max and normalize to target interval
    img_min = images.min()
    img_max = images.max()
    if img_max == img_min:
        # All pixels have same value, just shift to interval[0]
        return torch.full_like(images, interval[0])
    
    # Normalize to [0, 1] first, then scale to target interval
    normalized = (images - img_min) / (img_max - img_min)
    return normalized * (interval[1] - interval[0]) + interval[0]

def normalize_intrinsics(intrinsics: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """
    Normalizes intrinsics based on image resolution.
    Handles both single [3, 3] and batched [B, 3, 3] intrinsics.
    """
    H, W = image_size
    
    # Handle both batched and non-batched
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        was_unbatched = True
    else:
        was_unbatched = False
    
    if already_normalized(intrinsics, "intrinsics"):
        return intrinsics.squeeze(0) if was_unbatched else intrinsics
    
    # Clone to preserve device/dtype
    K = intrinsics.clone()
    K[:, 0, 0] /= W  # fx
    K[:, 1, 1] /= H  # fy
    K[:, 0, 2] /= W  # cx
    K[:, 1, 2] /= H  # cy
    
    return K.squeeze(0) if was_unbatched else K

def denormalize_images(images: torch.Tensor, interval: tuple[float, float] = (0, 1)) -> torch.Tensor:
    """
    Denormalizes images to original range.
    """
    return images * (interval[1] - interval[0]) + interval[0]

def denormalize_intrinsics(intrinsics: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """
    Denormalizes intrinsics to original range.
    Handles both single [3, 3] and batched [B, 3, 3] intrinsics.
    """
    H, W = image_size
    
    # Handle both batched and non-batched
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        was_unbatched = True
    else:
        was_unbatched = False
    
    # Clone to preserve device/dtype
    K = intrinsics.clone()
    K[:, 0, 0] *= W  # fx
    K[:, 1, 1] *= H  # fy
    K[:, 0, 2] *= W  # cx
    K[:, 1, 2] *= H  # cy
    
    return K.squeeze(0) if was_unbatched else K