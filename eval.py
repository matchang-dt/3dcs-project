"""
Evaluation script for MVSplat checkpoints.

Usage:
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt datasets=[re10k,acid]
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt save_images_dir=outputs/eval_images

With save_images_dir, each scene folder gets context_v*.png, rendered_v*.png, target_v*.png.
Replicate runs: set eval_seed and eval_num_workers: 0 in configs/eval.yaml (or CLI overrides).
"""
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure

from model import MVSplatConfig
from wrapper import MVSplatWrapper, LightningConfig
from decoder.decoder_cuda_splatting_gaussians import DecoderGaussianSplattingCUDACfg
from datasets.dataset_re10k import Re10kDataset
from datasets.dataset_acid import AcidDataset
from datasets.dataset_tnt import TanksAndTemplesDataset
from datasets.dataset_deepblending import DeepBlendingDataset
from utils.projection import make_proj_matrix


def set_eval_seed(seed: int) -> None:
    """Fix random, NumPy, and PyTorch RNGs so view sampling and model noise (if any) match across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_worker_init_fn(base_seed: int):
    def _fn(worker_id: int) -> None:
        set_eval_seed(base_seed + worker_id)

    return _fn


def compute_psnr(
    render: torch.Tensor,
    gt: torch.Tensor,
    mask_nonzero_only: bool = True,
    eps: float = 1e-6,
) -> float:
    """
    PSNR in dB (higher is better). If mask_nonzero_only=True, only pixels where
    the rendered image is non-zero (sum of channels > eps) are used when comparing to GT.
    """
    # Flatten to [N, C]
    C = render.shape[-3]
    render_flat = render.reshape(-1, C)
    gt_flat = gt.reshape(-1, C)
    if mask_nonzero_only:
        valid = (render_flat.abs().sum(dim=1) > eps)
        n_valid = valid.sum().item()
        if n_valid == 0:
            return 0.0
        render_flat = render_flat[valid]
        gt_flat = gt_flat[valid]
    mse = F.mse_loss(render_flat, gt_flat).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def collate_fn(batch):
    if not batch:
        return None
    ref_shape = batch[0]["target"]["images"].shape[0]
    batch = [b for b in batch if b["target"]["images"].shape[0] == ref_shape]
    if not batch:
        return None
    return {
        "context": {
            "images": torch.stack([b["context"]["images"] for b in batch]),
            "intrinsics": torch.stack([b["context"]["intrinsics"] for b in batch]),
            "extrinsics": torch.stack([b["context"]["extrinsics"] for b in batch]),
        },
        "target": {
            "images": torch.stack([b["target"]["images"] for b in batch]),
            "intrinsics": torch.stack([b["target"]["intrinsics"] for b in batch]),
            "extrinsics": torch.stack([b["target"]["extrinsics"] for b in batch]),
        },
        "scene_key": [b["scene_key"] for b in batch],
        "near_plane": batch[0].get("near_plane", 0.1),
        "far_plane": batch[0].get("far_plane", 100.0),
    }


def build_model(cfg, ckpt, device):
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    decoder_cfg = DecoderGaussianSplattingCUDACfg(name="cuda_gaussian_splatting")
    model_config = MVSplatConfig(
        image_size=cfg.image_size, hidden_dim=cfg.hidden_dim,
        swin_divisions=cfg.swin_divisions,
        cnn_dtype=dtype_map.get(cfg.cnn_dtype, torch.float32),
        transformer_dtype=dtype_map.get(cfg.transformer_dtype, torch.float32),
        pipeline_dtype=dtype_map.get(cfg.pipeline_dtype, torch.float32),
        feature_dim=cfg.feature_dim,
        gaussian_head_channels=cfg.gaussian_head_channels,
        opacity_start=cfg.opacity_start, opacity_end=cfg.opacity_end,
        opacity_warmup=cfg.opacity_warmup,
        sh_degree=cfg.sh_degree, scale_min=cfg.scale_min, scale_max=cfg.scale_max,
        gaussian_scale_pct=cfg.gaussian_scale_pct,
        gaussians_per_pixel=cfg.gaussians_per_pixel, num_surfaces=cfg.num_surfaces,
        decoder_cfg=decoder_cfg, dataset_cfg=None,
    )
    model = MVSplatWrapper.load_from_checkpoint(
        ckpt, model_config=model_config, lightning_config=LightningConfig(),
        map_location=device, strict=False, weights_only=False,
    )
    model.eval()
    model.to(device)
    return model


def timed_forward(model, batch, device):
    raw = model.model

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    batch = to_dev(batch)
    ctx_img = batch["context"]["images"]
    ctx_int = batch["context"]["intrinsics"]
    ctx_ext = batch["context"]["extrinsics"]
    tgt_int = batch["target"]["intrinsics"]
    tgt_ext = batch["target"]["extrinsics"]

    B, K, C, H, W = ctx_img.shape
    num_tgt = tgt_ext.shape[1]

    near = raw.cfg.near
    far = raw.cfg.far
    if isinstance(near, (int, float)):
        near = torch.tensor([near] * B * num_tgt, device=device).view(B, num_tgt)
    if isinstance(far, (int, float)):
        far = torch.tensor([far] * B * num_tgt, device=device).view(B, num_tgt)

    sync = lambda: torch.cuda.synchronize() if device.type == "cuda" else None

    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        # 1. Extractor: transformer features + CNN features (same as model.py)
        features, features_cnn = raw.extractor(ctx_img)
        features = features.to(raw.cfg.pipeline_dtype)       # [B, K, H//4, W//4, 128]
        features_cnn = features_cnn.to(raw.cfg.pipeline_dtype)  # [B, K, 128, H//4, W//4]
        with torch.autocast(device_type="cuda", enabled=False):
            proj = make_proj_matrix(ctx_ext, ctx_int)
        proj = proj.to(raw.cfg.pipeline_dtype)
        # 2. Cost volume: transformer features only
        cost_vol = raw.cost_volume_constructor(features=features, Ps=proj)
        # 3. Upsample concat(transformer, cnn) for depth and splat head
        features_flat = features.reshape(B * K, raw.cfg.feature_dim, H // 4, W // 4)
        features_cnn_flat = features_cnn.reshape(B * K, raw.cfg.feature_dim, H // 4, W // 4)
        upsampled_features_all = raw.feature_upsampler(
            torch.cat([features_flat, features_cnn_flat], dim=1)
        )  # [B*K, C, H, W]
        # 4. Depth estimator: cost_volume, images, upsampled features
        depth_maps, depth_conf = raw.depth_estimator(
            cost_volume=cost_vol, images=ctx_img, features=upsampled_features_all,
        )
        depth_maps = torch.clamp(depth_maps, min=near.min().item() + 1e-4, max=far.max().item() - 1e-4)
        head = getattr(raw, "splat_head", None) or getattr(raw, "gaussian_head", raw.splat_head)
        # 5. Splat head: features as [B, K, H, W, C]
        features_for_head = upsampled_features_all.reshape(
            B, K, raw.cfg.feature_dim, H, W
        ).permute(0, 1, 3, 4, 2)
        gaussians = head(
            depth_map=depth_maps, depth_conf=depth_conf, images=ctx_img, features=features_for_head,
            extrinsics=ctx_ext, intrinsics=ctx_int, global_step=0,
        )
        rays = H * W
        B_g, V_g = gaussians.means.shape[0], gaussians.means.shape[1]
        G = V_g * rays * raw.cfg.num_surfaces * raw.cfg.gaussians_per_pixel
        gaussians.means = gaussians.means.reshape(B_g, G, 3)
        gaussians.covariances = gaussians.covariances.reshape(B_g, G, 3, 3)
        gaussians.harmonics = gaussians.harmonics.reshape(B_g, G, 3, -1)
        gaussians.opacities = gaussians.opacities.reshape(B_g, G)
    sync(); gen_ms = (time.perf_counter() - t0) * 1000.0

    sync(); t1 = time.perf_counter()
    with torch.no_grad():
        rendered = raw.decoder(
            gaussians=gaussians, extrinsics=tgt_ext, intrinsics=tgt_int,
            near=near, far=far, image_shape=(H, W), depth_mode=None)
    sync(); render_ms = (time.perf_counter() - t1) * 1000.0

    return {"rendered_images": rendered["color"]}, gen_ms, render_ms


def evaluate_dataset(model, loader, name, device, lpips_fn, ssim_fn, save_dir, max_batches):
    scene_results = {}
    psnr_all, ssim_all, lpips_all, gen_times, render_times = [], [], [], [], []
    n = 0
    for batch in tqdm(loader, desc=name, unit="batch"):
        if batch is None:
            continue
        if max_batches is not None and n >= max_batches:
            break
        n += 1
        outputs, gen_ms, render_ms = timed_forward(model, batch, device)
        rendered = outputs["rendered_images"].clamp(0.0, 1.0)
        target = batch["target"]["images"].to(device).clamp(0.0, 1.0)
        B, V = rendered.shape[:2]
        flat_r = rendered.flatten(0, 1)
        flat_t = target.flatten(0, 1)
        psnr_all.append(compute_psnr(flat_r, flat_t))
        ssim_all.append(ssim_fn(flat_r, flat_t).item())
        lpips_all.append(lpips_fn(flat_r * 2 - 1, flat_t * 2 - 1).mean().item())
        gen_times.append(gen_ms)
        render_times.append(render_ms)
        for i in range(B):
            sk = str(batch["scene_key"][i]) if i < len(batch["scene_key"]) else f"scene_{n}_{i}"
            r_i = rendered[i].clamp(0, 1)
            t_i = target[i].clamp(0, 1)
            scene_results[sk] = {
                "psnr": round(compute_psnr(r_i, t_i), 4),
                "ssim": round(ssim_fn(r_i, t_i).item(), 4),
                "lpips": round(lpips_fn(r_i * 2 - 1, t_i * 2 - 1).mean().item(), 4),
                "gaussian_gen_time_ms": round(gen_ms / B, 2),
                "render_time_ms": round(render_ms / B, 2),
            }
            if save_dir:
                out = Path(save_dir) / name / sk
                out.mkdir(parents=True, exist_ok=True)
                c_i = batch["context"]["images"][i].detach().cpu().clamp(0.0, 1.0)
                for k in range(c_i.shape[0]):
                    torchvision.utils.save_image(c_i[k], out / f"context_v{k:02d}.png")
                for v in range(r_i.shape[0]):
                    torchvision.utils.save_image(r_i[v].cpu(), out / f"rendered_v{v:02d}.png")
                    torchvision.utils.save_image(t_i[v].cpu(), out / f"target_v{v:02d}.png")

    def mean(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
    return {
        "psnr": mean(psnr_all), "ssim": mean(ssim_all), "lpips": mean(lpips_all),
        "gaussian_gen_time_ms": mean(gen_times), "render_time_ms": mean(render_times),
        "num_batches": n, "scenes": scene_results,
    }


def _ns(name, root, stage, ni, nt, sz):
    return SimpleNamespace(
        name=name, data_root=root, stage=stage,
        num_input_views=ni, num_target_views=nt,
        target_image_size=sz, max_train_steps=0, normalize_scene=False,
    )


def build_eval_loaders(cfg, eval_seed: int):
    wanted = list(cfg.datasets)
    ni, nt, sz = int(cfg.num_input_views), int(cfg.num_target_views), int(cfg.image_size)
    bs = int(cfg.batch_size)
    nw = OmegaConf.select(cfg, "eval_num_workers", default=None)
    if nw is None:
        nw = int(cfg.num_workers)
    else:
        nw = int(nw)
    worker_kw = {}
    if nw > 0:
        worker_kw["worker_init_fn"] = _make_worker_init_fn(eval_seed)
    loaders = {}
    if "re10k" in wanted:
        loaders["re10k"] = DataLoader(
            Re10kDataset(cfg=_ns("re10k", cfg.re10k_data_root, "test", ni, nt, sz)),
            batch_size=bs, num_workers=nw, collate_fn=collate_fn, pin_memory=True, **worker_kw)
    if "acid" in wanted:
        loaders["acid"] = DataLoader(
            AcidDataset(cfg=_ns("acid", cfg.acid_data_root, "test", ni, nt, sz)),
            batch_size=bs, num_workers=nw, collate_fn=collate_fn, pin_memory=True, **worker_kw)
    if "tnt" in wanted:
        nt_tnt = -1 if nt <= 0 else nt
        loaders["tnt"] = DataLoader(
            TanksAndTemplesDataset(data_root=cfg.tnt_data_root, stage="test",
                num_input_views=ni, num_target_views=nt_tnt,
                target_image_size=sz, max_train_steps=0),
            batch_size=1, num_workers=nw, collate_fn=collate_fn, pin_memory=True, **worker_kw)
    if "deepblending" in wanted:
        nt_db = -1 if nt <= 0 else nt
        loaders["deepblending"] = DataLoader(
            DeepBlendingDataset(data_root=cfg.deepblending_data_root, stage="test",
                num_input_views=ni, num_target_views=nt_db,
                target_image_size=sz, max_train_steps=0),
            batch_size=1, num_workers=nw, collate_fn=collate_fn, pin_memory=True, **worker_kw)
    return loaders


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    ckpt = str(cfg.checkpoint_path)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    eval_seed = int(OmegaConf.select(cfg, "eval_seed", default=42))
    set_eval_seed(eval_seed)
    print(f"eval_seed={eval_seed} (same checkpoint + seed + eval_num_workers reproduces view sampling)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg, ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    lpips_fn = LPIPS(net="vgg").to(device).eval()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    loaders = build_eval_loaders(cfg, eval_seed=eval_seed)
    if not loaders:
        raise ValueError(f"No datasets to evaluate. datasets={list(cfg.datasets)}")

    max_b = OmegaConf.select(cfg, "max_batches_per_dataset")
    save_dir = OmegaConf.select(cfg, "save_images_dir")
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": ckpt,
        "num_parameters": n_params,
        "eval_seed": eval_seed,
        "eval_num_workers": OmegaConf.select(cfg, "eval_num_workers", default=None),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "datasets": {},
    }

    for ds_name, loader in loaders.items():
        print(f"\n--- {ds_name} ---")
        summary = evaluate_dataset(
            model=model, loader=loader, name=ds_name, device=device,
            lpips_fn=lpips_fn, ssim_fn=ssim_fn,
            save_dir=save_dir, max_batches=max_b,
        )
        results["datasets"][ds_name] = summary
        print(
            f"PSNR={summary['psnr']:.4f}  SSIM={summary['ssim']:.4f}  "
            f"LPIPS={summary['lpips']:.4f}  "
            f"GaussGen={summary['gaussian_gen_time_ms']:.1f}ms  "
            f"Render={summary['render_time_ms']:.1f}ms"
        )

    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {out}")


if __name__ == "__main__":
    main()
