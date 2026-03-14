"""
Hydra + Lightning training script for MVSplat.

Usage:
    python train.py                         # Default: RE10K dataset
    python train.py dataset=acid_train      # Use ACID dataset
    python train.py batch_size=4            # Override training params
"""
import os
from typing import Any

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from model import MVSplatConfig
from wrapper import MVSplatWrapper, LightningConfig
from decoder.decoder_cuda_splatting_gaussians import DecoderGaussianSplattingCUDACfg
from datasets.dataset import DatasetCfg, DATASETS


class UpdateDatasetGlobalStepCallback(Callback):
    """Updates the train dataset's current_step for baseline-expansion view sampling."""

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int) -> None:
        dl = trainer.train_dataloader
        if dl is None:
            return
        # Single DataLoader (our case) or CombinedLoader with .loaders
        loaders = getattr(dl, "loaders", [dl])
        if not isinstance(loaders, list):
            loaders = [loaders]
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_training_step"):
                dataset.set_training_step(trainer.global_step)
                break

def create_model_from_hydra_config(cfg: DictConfig) -> MVSplatWrapper:
    """
    Create MVSplat Lightning wrapper from Hydra config.
    
    Args:
        cfg: Hydra DictConfig with model and training parameters
        
    Returns:
        MVSplatWrapper ready for training
    """
    # Create decoder config
    decoder_cfg = DecoderGaussianSplattingCUDACfg(name="cuda_gaussian_splatting")
    
    # Convert dtype strings to torch dtypes
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    
    # Create model config
    model_config = MVSplatConfig(
        image_size=cfg.image_size,
        hidden_dim=cfg.hidden_dim,
        swin_divisions=cfg.swin_divisions,
        cnn_dtype=dtype_map.get(cfg.cnn_dtype, torch.float32),
        transformer_dtype=dtype_map.get(cfg.transformer_dtype, torch.float32),
        pipeline_dtype=dtype_map.get(cfg.pipeline_dtype, torch.float32),
        feature_dim=cfg.feature_dim,
        gaussian_head_channels=cfg.gaussian_head_channels,
        opacity_start=cfg.opacity_start,
        opacity_end=cfg.opacity_end,
        opacity_warmup=cfg.opacity_warmup,
        sh_degree=cfg.sh_degree,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
        gaussian_scale_pct=cfg.gaussian_scale_pct,
        gaussians_per_pixel=cfg.gaussians_per_pixel,
        num_surfaces=cfg.num_surfaces,
        decoder_cfg=decoder_cfg,
        dataset_cfg=None,
    )
    
    # Create Lightning config
    lightning_config = LightningConfig(
        optimizer_name=cfg.optimizer.name,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        scheduler_name=cfg.scheduler.name,
        scheduler_T_max=cfg.scheduler.T_max,
        scheduler_eta_min=cfg.scheduler.eta_min,
        warmup_steps=cfg.scheduler.warmup_steps,
        rgb_loss_weight=cfg.loss.rgb_loss_weight,
        lpips_loss_weight=cfg.loss.lpips_loss_weight,
        log_images_every_n_steps=cfg.log_images_every_n_steps,
        val_check_interval=cfg.val_check_interval,
    )
    
    # Create and return Lightning wrapper
    return MVSplatWrapper(model_config, lightning_config)


def collate_fn(batch):
    """Custom collate function for IterableDataset."""
    # batch is a list of dicts from the dataset
    # Each dict has 'context' and 'target' subdicts
    
    if len(batch) == 0:
        return None
    
    # Filter batch for consistent target view counts
    # Sometimes view sampler returns different number of target views (e.g. at end of epoch or edge cases)
    # We enforce that all items in the batch must have the same number of target views as the first item
    if len(batch) > 0:
        ref_shape = batch[0]['target']['images'].shape[0]
        valid_indices = [i for i, item in enumerate(batch) if item['target']['images'].shape[0] == ref_shape]
        
        if len(valid_indices) < len(batch):
            # print(f"Warning: Dropped {len(batch) - len(valid_indices)} items with inconsistent target views")
            batch = [batch[i] for i in valid_indices]
            
    if len(batch) == 0:
        return None
    
    # Stack context views
    context_images = torch.stack([item['context']['images'] for item in batch])
    context_intrinsics = torch.stack([item['context']['intrinsics'] for item in batch])
    context_extrinsics = torch.stack([item['context']['extrinsics'] for item in batch])
    
    # Stack target views
    target_images = torch.stack([item['target']['images'] for item in batch])
    target_intrinsics = torch.stack([item['target']['intrinsics'] for item in batch])
    target_extrinsics = torch.stack([item['target']['extrinsics'] for item in batch])
    
    return {
        'context': {
            'images': context_images,
            'intrinsics': context_intrinsics,
            'extrinsics': context_extrinsics,
        },
        'target': {
            'images': target_images,
            'intrinsics': target_intrinsics,
            'extrinsics': target_extrinsics,
        },
        'scene_key': [item['scene_key'] for item in batch],
        'near_plane': batch[0].get('near_plane', 0.1),
        'far_plane': batch[0].get('far_plane', 100.0),
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    
    # Print config
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set seed
    L.seed_everything(cfg.seed, workers=True)

    print(f"Config: {cfg}")
    
    # Create datasets
    print("\nCreating datasets...")
    print(f"Dataset: {cfg.dataset.name} (data_root={cfg.dataset.data_root})")
    train_dataset = DATASETS[cfg.dataset.name](cfg=cfg.dataset)
    # Validation/test dataset: same dataset class, config merged with val_dataset overrides (e.g. stage: test)
    val_dataset_cfg = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True)),
        OmegaConf.create(OmegaConf.to_container(cfg.get("val_dataset", {}), resolve=True)),
    )
    val_dataset = DATASETS[cfg.dataset.name](cfg=val_dataset_cfg)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Train dataset: {cfg.dataset.name} (stage={cfg.dataset.get('stage', 'train')})")
    print(f"Val dataset: {cfg.dataset.name} (stage={val_dataset_cfg.get('stage', 'test')})")
    
    # Create model
    print("\nCreating model...")
    model = create_model_from_hydra_config(cfg)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        every_n_train_steps=cfg.checkpoint.save_every_n_train_steps,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    step_callback = UpdateDatasetGlobalStepCallback()

    # Setup logger
    logger = WandbLogger(
        save_dir='logs/',
        name=cfg.experiment_name,
    )
    
    # Create trainer
    limit_val_batches = cfg.get("limit_val_batches", None)
    trainer = L.Trainer(
        max_steps=cfg.max_steps,
        accelerator='auto',
        devices=cfg.num_gpus,
        precision=cfg.precision,
        callbacks=[checkpoint_callback, lr_monitor, step_callback],
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=cfg.gradient_clip_val,
        deterministic=False,
    )
    
    # Train (validation uses RE10K test set via val_dataset.stage: test)
    print("\nStarting training...")
    print(f"Logs will be saved to: {logger.log_dir}")
    print(f"Checkpoints will be saved to: {cfg.checkpoint.dirpath}")

    if cfg.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
    trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=cfg.resume_from_checkpoint, weights_only=False)

    print("\nTraining complete!")


if __name__ == '__main__':
    import os
    os.environ.setdefault("WANDB_MODE", "online")
    main()
