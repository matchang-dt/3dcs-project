"""
Evaluation script for MVSplat checkpoints.

Usage:
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt datasets=[re10k,acid]
    python eval.py checkpoint_path=/path/to/checkpoint.ckpt save_images_dir=outputs/eval_images

View sampling for test sets is deterministic (eval_seed) so results are reproducible.
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
from datasets.dataset_dtu import DTUDataset
from utils.projection import make_proj_matrix


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Seed RNG in DataLoader workers so view sampling is deterministic."""
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def compute_psnr(render: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(render, gt).item()
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
        ckpt,
        model_config=model_config,
        lightning_config=LightningConfig(),
        map_location=device,
        strict=False,
        weights_only=False,
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
        features = raw.extractor(ctx_img).to(raw.cfg.pipeline_dtype)
        with torch.autocast(device_type="cuda", enabled=False):
            proj = make_proj_matrix(ctx_ext, ctx_int)
        proj = proj.to(raw.cfg.pipeline_dtype)
        cost_vol = raw.cost_volume_constructor(features=features, Ps=proj)
        depth_maps, depth_conf = raw.depth_estimator(
            cost_volume=cost_vol, images=ctx_img, features=features)
        depth_maps = torch.clamp(depth_maps, min=near.min() + 1e-4, max=far.max() - 1e-4)
        head = getattr(raw, "splat_head", None) or raw.gaussian_head
        gaussians = head(
            depth_map=depth_maps, depth_conf=depth_conf, images=ctx_img, features=features,
            extrinsics=ctx_ext, intrinsics=ctx_int, global_step=0)
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
                for v in range(r_i.shape[0]):
                    torchvision.utils.save_image(r_i[v], out / f"rendered_v{v:02d}.png")
                    torchvision.utils.save_image(t_i[v], out / f"target_v{v:02d}.png")

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


def build_eval_loaders(cfg):
    wanted = list(cfg.datasets)
    ni, nt, sz = int(cfg.num_input_views), int(cfg.num_target_views), int(cfg.image_size)
    nw, bs = int(cfg.num_workers), int(cfg.batch_size)
    eval_seed = int(OmegaConf.select(cfg, "eval_seed", default=42))
    seed_fn = lambda worker_id: _seed_worker(worker_id, eval_seed)

    loaders = {}
    if "re10k" in wanted:
        loaders["re10k"] = DataLoader(
            Re10kDataset(cfg=_ns("re10k", cfg.re10k_data_root, "test", ni, nt, sz)),
            batch_size=bs, num_workers=nw, collate_fn=collate_fn, pin_memory=True,
            worker_init_fn=seed_fn,
        )
    if "acid" in wanted:
        loaders["acid"] = DataLoader(
            AcidDataset(cfg=_ns("acid", cfg.acid_data_root, "test", ni, nt, sz)),
            batch_size=bs, num_workers=nw, collate_fn=collate_fn, pin_memory=True,
            worker_init_fn=seed_fn,
        )
    if "tnt" in wanted:
        nt_tnt = -1 if nt <= 0 else nt
        loaders["tnt"] = DataLoader(
            TanksAndTemplesDataset(data_root=cfg.tnt_data_root, stage="test",
                num_input_views=ni, num_target_views=nt_tnt,
                target_image_size=sz, max_train_steps=0),
            batch_size=1, num_workers=nw, collate_fn=collate_fn, pin_memory=True,
            worker_init_fn=seed_fn,
        )
    if "deepblending" in wanted:
        nt_db = -1 if nt <= 0 else nt
        loaders["deepblending"] = DataLoader(
            DeepBlendingDataset(data_root=cfg.deepblending_data_root, stage="test",
                num_input_views=ni, num_target_views=nt_db,
                target_image_size=sz, max_train_steps=0),
            batch_size=1, num_workers=nw, collate_fn=collate_fn, pin_memory=True,
            worker_init_fn=seed_fn,
        )
    if "dtu" in wanted:
        nt_dtu = -1 if nt <= 0 else nt
        dtu_light = int(OmegaConf.select(cfg, "dtu_light_idx", default=3))
        loaders["dtu"] = DataLoader(
            DTUDataset(
                data_root=cfg.dtu_data_root,
                stage="test",
                num_input_views=ni,
                num_target_views=nt_dtu,
                target_image_size=(sz, sz),
                max_train_steps=0,
                light_idx=dtu_light,
            ),
            batch_size=1,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=seed_fn,
        )
    return loaders


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Fixed RNG for reproducible view sampling across runs
    eval_seed = int(OmegaConf.select(cfg, "eval_seed", default=42))
    random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    ckpt = str(cfg.checkpoint_path)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg, ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    lpips_fn = LPIPS(net="vgg").to(device).eval()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    loaders = build_eval_loaders(cfg)
    if not loaders:
        raise ValueError(f"No datasets to evaluate. datasets={list(cfg.datasets)}")

    max_b = OmegaConf.select(cfg, "max_batches_per_dataset")
    save_dir = OmegaConf.select(cfg, "save_images_dir")
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": ckpt, "num_parameters": n_params,
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
