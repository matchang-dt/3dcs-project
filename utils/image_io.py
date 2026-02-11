"""
Save tensor images to disk (PNG/JPEG).

Tensors are expected in [C, H, W] or [B, C, H, W], values in [0, 1] (or will be clamped).
"""
import os
from pathlib import Path

import torch


def save_image_tensor(
    tensor: torch.Tensor,
    path: str | Path,
    clamp: bool = True,
    scale_depth: bool = False,
) -> None:
    """
    Save a single image tensor to disk.

    Args:
        tensor: [C, H, W] or [3, H, W] for RGB; [1, H, W] or [H, W] for grayscale/depth.
        path: Output file path (.png or .jpg).
        clamp: If True, clamp values to [0, 1] before saving.
        scale_depth: If True, treat as depth and normalize to [0,1] for visualization.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tensor.dim() == 3:
        tensor = tensor.detach().cpu()
    elif tensor.dim() == 2:
        tensor = tensor.detach().cpu().unsqueeze(0)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")

    if scale_depth:
        tensor = tensor.float()
        mn, mx = tensor.min().item(), tensor.max().item()
        if mx > mn:
            tensor = (tensor - mn) / (mx - mn)
        else:
            tensor = torch.zeros_like(tensor)
    elif clamp:
        tensor = tensor.float().clamp(0.0, 1.0)

    # [C, H, W] -> [H, W, C] for PIL, uint8
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # [H, W]
    if tensor.dim() == 2:
        arr = (tensor.numpy() * 255).astype("uint8")
    else:
        tensor = tensor.permute(1, 2, 0)  # [H, W, C]
        arr = (tensor.numpy() * 255).astype("uint8")

    try:
        from PIL import Image
    except ImportError:
        import torchvision
        torchvision.utils.save_image(
            tensor.permute(2, 0, 1) if tensor.dim() == 3 else tensor.unsqueeze(0),
            str(path),
        )
        return

    img = Image.fromarray(arr)
    img.save(str(path))


def save_batch_images(
    rendered: torch.Tensor,
    target: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
    depth_rendered: torch.Tensor | None = None,
    out_dir: str | Path = "outputs/images",
    step: int = 0,
    batch_idx: int = 0,
    prefix: str = "",
) -> list[Path]:
    """
    Save rendered/target/depth from a batch to disk.

    Args:
        rendered: [B, V, 3, H, W] or [V, 3, H, W] – rendered RGB.
        target: Optional [B, V, 3, H, W] – target RGB.
        depth: Optional [B, V, H, W] – estimated depth (context views).
        depth_rendered: Optional [B, V, H, W] – rendered depth (target views).
        out_dir: Directory to write images.
        step: Global step or sample id.
        batch_idx: Batch index.
        prefix: Optional prefix for filenames.

    Returns:
        List of saved file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def _save(name: str, t: torch.Tensor, scale_depth: bool = False) -> None:
        t = t.detach().cpu().float().clamp(0.0, 1.0) if not scale_depth else t.detach().cpu().float()
        p = out_dir / f"{prefix}{name}_step{step}_b{batch_idx}.png"
        save_image_tensor(t, p, clamp=not scale_depth, scale_depth=scale_depth)
        saved.append(p)

    # Rendered: first batch item, first target view
    if rendered.ndim == 5:
        _save("rendered", rendered[0, 0])
        # Optionally save all target views
        for v in range(rendered.shape[1]):
            _save(f"rendered_v{v}", rendered[0, v])
    else:
        _save("rendered", rendered[0] if rendered.dim() == 4 else rendered)

    if target is not None:
        if target.ndim == 5:
            _save("target", target[0, 0])
        else:
            _save("target", target[0] if target.dim() == 4 else target)

    if depth is not None:
        if depth.ndim == 4:
            _save("depth_est", depth[0, 0], scale_depth=True)
        else:
            _save("depth_est", depth[0] if depth.dim() == 3 else depth, scale_depth=True)

    if depth_rendered is not None:
        if depth_rendered.ndim == 4:
            _save("depth_rendered", depth_rendered[0, 0], scale_depth=True)
        else:
            _save("depth_rendered", depth_rendered[0], scale_depth=True)

    return saved
