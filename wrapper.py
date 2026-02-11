"""
PyTorch Lightning wrapper for MVSplat model.

This module provides a clean separation between the core MVSplat nn.Module
and the Lightning training logic. It can be used standalone or with Hydra configs.
"""
import torch
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from torchvision.utils import make_grid

from model import MVSplat, MVSplatConfig
from lpips import LPIPS
from lightning.pytorch.loggers import WandbLogger


def _apply_jet_cmap(x: torch.Tensor) -> torch.Tensor:
    """Apply jet colormap (blue=low, red=high) to a 2D tensor. Returns [3, H, W] RGB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_np = x.detach().cpu().float().numpy()
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    cm = plt.get_cmap("jet")
    rgb = cm(x_np)[:, :, :3]  # [H, W, 3]
    out = torch.from_numpy(rgb).float().permute(2, 0, 1).to(device=x.device)
    return out

@dataclass
class LightningConfig:
    """Configuration for Lightning training wrapper."""
    # Optimizer params
    optimizer_name: str = 'adam'
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    
    # Scheduler params
    scheduler_name: Optional[str] = 'cosine'
    scheduler_T_max: int = 100000
    scheduler_eta_min: float = 1e-6
    warmup_steps: int = 0
    
    # Loss weights
    rgb_loss_weight: float = 1.0
    lpips_loss_weight: float = 0.05
    
    # Logging
    log_images_every_n_steps: int = 1000


class MVSplatWrapper(L.LightningModule):
    """
    PyTorch Lightning wrapper for MVSplat model.
    
    This class handles:
    - Training and validation steps
    - Loss computation
    - Optimizer and scheduler configuration
    - Logging metrics and images to TensorBoard
    - Checkpointing
    
    Usage:
        # Create model config
        model_config = MVSplatConfig(image_size=256, ...)
        
        # Create Lightning wrapper
        lightning_config = LightningConfig(learning_rate=1e-4, ...)
        wrapper = MVSplatLightningWrapper(model_config, lightning_config)
        
        # Train with PyTorch Lightning Trainer
        trainer = L.Trainer(max_steps=100000, ...)
        trainer.fit(wrapper, train_dataloader)
    """
    
    def __init__(
        self, 
        model_config: MVSplatConfig,
        lightning_config: Optional[LightningConfig] = None
    ):
        """
        Initialize Lightning wrapper.
        
        Args:
            model_config: Configuration for the MVSplat model
            lightning_config: Configuration for training (optimizer, scheduler, etc.)
                            If None, uses default LightningConfig.
        """
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Store configs
        self.model_config = model_config
        self.lightning_config = lightning_config or LightningConfig()
        self.model = MVSplat(model_config)
        self.lpips = LPIPS(net='vgg')
        for p in self.lpips.parameters(): # freeze LPIPS
            p.requires_grad = False
        
    def forward(self, batch: Dict[str, Any], render_depth: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing context and target data
            render_depth: Whether to render depth maps
            
        Returns:
            Dictionary with rendered images, depth maps, etc.
        """
        return self.model(batch, global_step=self.global_step, render_depth=render_depth)
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            outputs: Model outputs (rendered_images, depth_maps, etc.)
            batch: Input batch with target images
            
        Returns:
            Dictionary with 'loss' and individual loss components
        """
        rendered = outputs['rendered_images']  # [B, V_target, 3, H, W]
        target = batch['target']['images']  # [B, V_target, 3, H, W] or [V_target, 3, H, W]

        if target.ndim == 4:
            target = target.unsqueeze(0)

        rgb_loss = torch.mean(F.mse_loss(rendered.flatten(0, 1), target.flatten(0, 1)))
        lpips_loss = torch.mean(self.lpips(rendered.flatten(0, 1), target.flatten(0, 1))) * self.lightning_config.lpips_loss_weight
        return {
            'loss': rgb_loss + lpips_loss,
            'rgb_loss': rgb_loss,
            'lpips_loss': lpips_loss,
        }
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        """
        Training step - called by Lightning for each batch.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor for backpropagation
        """
        if batch is None:
            return None
            
        # model out
        outputs = self(batch, render_depth=False)
        losses = self.compute_loss(outputs, batch)
        
        # log losses to logger
        for name, value in losses.items():
            self.log(
                f'train/{name}', 
                value, 
                prog_bar=(name == 'loss'),
                sync_dist=True,
                on_step=True,
                on_epoch=False,
            )
        
        # log learning rate to logger
        opt = self.optimizers()
        if opt is not None:
            self.log(
                'train/lr', 
                opt.param_groups[0]['lr'], 
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )

        # periodically log images during training
        if self.global_step % self.lightning_config.log_images_every_n_steps == 0:
            self.log_images(batch, outputs)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        """
        Validation step - called by Lightning for each validation batch.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        if batch is None:
            return None
            
        outputs = self(batch, render_depth=True)
        losses = self.compute_loss(outputs, batch)
        
        # log losses to logger
        for name, value in losses.items():
            self.log(
                f'val/{name}', 
                value, 
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
        
        # periodically log images
        if self.global_step % self.lightning_config.log_images_every_n_steps == 0:
            self.log_images(batch, outputs)
        
        return losses['loss']
    
    def log_images(self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]):
        """
        Log images for the first batch: stitched grids of renders, targets, and context (input) images.
        Depth and diff grids use jet colormap (blue=low, red=high). All images are [3, H, W].
        """
        if self.logger is None:
            return

        def ensure_3ch(t: torch.Tensor) -> torch.Tensor:
            """Ensure tensor is [3, H, W] for logging."""
            if t.ndim == 2:
                t = t.unsqueeze(0).expand(3, -1, -1)
            return t.clamp(0.0, 1.0).float()

        try:
            rendered = outputs["rendered_images"]  # [B, V, 3, H, W] or [V, 3, H, W]
            target = batch["target"]["images"]
            if target.ndim == 4:
                target = target.unsqueeze(0)
            # First batch only
            if rendered.ndim == 5:
                rendered_b0 = rendered[0]  # [V, 3, H, W]
                target_b0 = target[0]      # [V, 3, H, W]
            else:
                rendered_b0 = rendered
                target_b0 = target

            num_views = rendered_b0.shape[0]
            if num_views == 0:
                return

            # Context (input) images: [B, V_ctx, 3, H, W]
            context_images = batch["context"]["images"]
            if context_images.ndim == 5:
                context_b0 = context_images[0]  # [V_ctx, 3, H, W]
            else:
                context_b0 = context_images.unsqueeze(0) if context_images.ndim == 4 else context_images
            num_context = context_b0.shape[0]

            # Stitch into grids (nrow caps columns for a compact layout)
            nrow = min(4, max(1, num_views))
            rendered_grid = make_grid(
                rendered_b0.clamp(0.0, 1.0).float(),
                nrow=nrow,
                padding=4,
                normalize=False,
            )
            target_grid = make_grid(
                target_b0.clamp(0.0, 1.0).float(),
                nrow=nrow,
                padding=4,
                normalize=False,
            )
            context_nrow = min(4, max(1, num_context))
            context_grid = make_grid(
                context_b0.clamp(0.0, 1.0).float(),
                nrow=context_nrow,
                padding=4,
                normalize=False,
            )

            # Depth only when we have rendered_depth
            has_depth = "rendered_depth" in outputs and outputs["rendered_depth"] is not None
            depth_grid = None
            if has_depth:
                d = outputs["rendered_depth"]
                depth_b0 = d[0] if d.ndim == 4 else d  # [V, H, W]
                depth_global_max = depth_b0.max() + 1e-6
                depth_vis = (depth_b0 / depth_global_max).clamp(0, 1)
                depth_jet_list = [_apply_jet_cmap(depth_vis[v]) for v in range(depth_vis.shape[0])]
                depth_stack = torch.stack(depth_jet_list, dim=0)  # [V, 3, H, W]
                depth_grid = make_grid(depth_stack, nrow=nrow, padding=4, normalize=False)

            # Diff grid (rendered vs target)
            diff_list = []
            for v in range(num_views):
                r_v = rendered_b0[v].clamp(0, 1).float()
                t_v = target_b0[v].clamp(0, 1).float()
                diff_v = (r_v - t_v).abs().mean(dim=0)
                diff_list.append(_apply_jet_cmap(diff_v))
            diff_stack = torch.stack(diff_list, dim=0)  # [V, 3, H, W]
            diff_grid = make_grid(diff_stack, nrow=nrow, padding=4, normalize=False)

            # Log stitched grids
            step = self.global_step
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image(
                    key="val/rendered_grid",
                    images=[rendered_grid.cpu().permute(1, 2, 0).numpy()],
                    caption=["rendered (all target views)"],
                )
                self.logger.log_image(
                    key="val/target_grid",
                    images=[target_grid.cpu().permute(1, 2, 0).numpy()],
                    caption=["target (all target views)"],
                )
                self.logger.log_image(
                    key="val/context_grid",
                    images=[context_grid.cpu().permute(1, 2, 0).numpy()],
                    caption=["context (input views)"],
                )
                self.logger.log_image(
                    key="val/diff_grid",
                    images=[diff_grid.cpu().permute(1, 2, 0).numpy()],
                    caption=["|rendered - target| (jet)"],
                )
                if depth_grid is not None:
                    self.logger.log_image(
                        key="val/depth_grid",
                        images=[depth_grid.cpu().permute(1, 2, 0).numpy()],
                        caption=["rendered depth (jet)"],
                    )
            elif hasattr(self.logger.experiment, "add_image"):
                self.logger.experiment.add_image("val/rendered_grid", rendered_grid, step)
                self.logger.experiment.add_image("val/target_grid", target_grid, step)
                self.logger.experiment.add_image("val/context_grid", context_grid, step)
                self.logger.experiment.add_image("val/diff_grid", diff_grid, step)
                if depth_grid is not None:
                    self.logger.experiment.add_image("val/depth_grid", depth_grid, step)
        except Exception as e:
            print(f"Warning: Error logging images: {e}")
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer or dict with optimizer and scheduler
        """
        cfg = self.lightning_config
        
        # Create optimizer
        if cfg.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )
        elif cfg.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")
        
        # Return just optimizer if no scheduler
        if cfg.scheduler_name is None:
            return optimizer
        
        # Create learning rate scheduler
        if cfg.scheduler_name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.scheduler_T_max,
                eta_min=cfg.scheduler_eta_min,
            )
        elif cfg.scheduler_name.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30000,
                gamma=0.1,
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler_name}")
        
        # Add warmup if specified
        if cfg.warmup_steps > 0:
            def lr_lambda(step):
                if step < cfg.warmup_steps:
                    return step / cfg.warmup_steps
                return 1.0
            
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': torch.optim.lr_scheduler.ChainedScheduler(
                        [warmup_scheduler, scheduler]
                    ),
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
    
    def on_train_start(self):
        """Called when training starts."""
        print(f"\n{'='*80}")
        print(f"Starting MVSplat Training")
        print(f"{'='*80}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Learning rate: {self.lightning_config.learning_rate}")
        print(f"Optimizer: {self.lightning_config.optimizer_name}")
        print(f"Scheduler: {self.lightning_config.scheduler_name}")
        print(f"{'='*80}\n")
