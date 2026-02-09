"""
PyTorch Lightning wrapper for MVSplat model.

This module provides a clean separation between the core MVSplat nn.Module
and the Lightning training logic. It can be used standalone or with Hydra configs.
"""
import torch
import torch.nn.functional as F
import lightning as L
from typing import Dict, Any, Optional
from dataclasses import dataclass

from model import MVSplat, MVSplatConfig
from lpips import LPIPS


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
        self.lpips = LPIPS(net_type='vgg').to(self.model.device)
        
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

        rgb_loss = F.mse_loss(rendered, target)
        lpips_loss = self.lpips(rendered, target)
        return {
            'loss': rgb_loss + lpips_loss,
            'rgb_loss': rgb_loss,
            'lpips_loss': lpips_loss,
        }
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step - called by Lightning for each batch.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor for backpropagation
        """
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
        if self.global_step % self.log_images_every_n_steps == 0:
            self.log_images(batch, outputs)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Validation step - called by Lightning for each validation batch.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
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
        if self.global_step % self.log_images_every_n_steps == 0:
            self.log_images(batch, outputs)
        
        return losses['loss']
    
    def log_images(self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]):
        """
        Log sample images to TensorBoard.
        
        Args:
            batch: Input batch
            outputs: Model outputs
        """
        if self.logger is None:
            return
        
        try:
            # Get first sample in batch
            rendered = outputs['rendered_images']
            if rendered.ndim == 5:  # [B, V, 3, H, W]
                rendered = rendered[0, 0]  # Take first batch, first view
            else:  # [V, 3, H, W]
                rendered = rendered[0]
            
            target = batch['target']['images']
            if target.ndim == 5:
                target = target[0, 0]
            elif target.ndim == 4:
                target = target[0]
            
            # Get depth map if available
            if 'rendered_depth' in outputs and outputs['rendered_depth'] is not None:
                depth = outputs['rendered_depth']
                if depth.ndim == 4:  # [B, V, H, W]
                    depth = depth[0, 0]
                elif depth.ndim == 3:  # [V, H, W]
                    depth = depth[0]
                
                # Normalize depth for visualization
                depth_vis = depth / (depth.max() + 1e-6)
                depth_vis = depth_vis.unsqueeze(0) if depth_vis.ndim == 2 else depth_vis
            else:
                # Use depth_maps from estimator as fallback
                depth = outputs['depth_maps']
                if depth.ndim == 4:  # [B, V, H, W]
                    depth = depth[0, 0]
                else:  # [V, H, W]
                    depth = depth[0]
                depth_vis = depth / (depth.max() + 1e-6)
            
            # Clamp rendered and target to [0, 1] for visualization
            rendered = torch.clamp(rendered, 0, 1)
            target = torch.clamp(target, 0, 1)
            
            # Log to TensorBoard
            self.logger.experiment.add_image(
                'val/rendered', 
                rendered, 
                self.global_step
            )
            self.logger.experiment.add_image(
                'val/target', 
                target, 
                self.global_step
            )
            self.logger.experiment.add_image(
                'val/depth', 
                depth_vis, 
                self.global_step
            )
            
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