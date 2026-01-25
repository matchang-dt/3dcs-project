"""
ACID Dataset

Each shard (.torch file) consists of:
- url: (str) scene identifier
- timestamps: (torch.Tensor) shape [V] timestamps in ms
- cameras: (torch.Tensor) shape [V, 18] format: [fx, fy, cx, cy, 0, 0, row major 3x4 flattened]
    Note: intrinsics (fx, fy, cx, cy) are normalized to [0, 1]
- images: (list[torch.Tensor]) length V, each is compressed JPEG as 1D tensor
- key: (str) hash identifying the scene (in provided index.json file)
"""

import os
import torch
import torchvision.transforms.functional as TF
from dataclasses import dataclass
from torch.utils.data import IterableDataset
from typing import List, Optional, Tuple
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

from .view_sampler.view_sampler import ViewSet, ViewSampler, ViewSamplerDefault

@dataclass
class AcidDatasetCfg:
    data_root: str = "/workspace/re10kvol/acid"
    stage: str = "train"
    num_input_views: int = 2
    num_target_views: int = 4
    target_image_size: Tuple[int, int] = (256, 256)
    max_train_steps: int = 300000
    view_sampler: ViewSampler = None

class AcidDataset(IterableDataset):    
    def __init__(
        self,
        data_root: str = "/workspace/re10kvol/acid",
        stage: str = "train",
        num_input_views: int = 2,
        num_target_views: int = 4,
        target_image_size: Tuple[int, int] = (256, 256),
        max_train_steps: int = 300000,
        view_sampler: ViewSampler = None,
    ):
        """
        Args:
            data_root: Root directory containing {train,test,validation}/ subdirectories
            stage: One of 'train', 'test', 'validation'
            num_input_views: Number of context/input views
            num_target_views: Number of target views for supervision
            target_image_size: Resize images to this size (height=width)
            max_train_steps: Maximum training steps (for baseline expansion schedule)
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.stage = stage
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.target_image_size = target_image_size
        self.max_train_steps = max_train_steps
        
        # load shards
        stage_dir = self.data_root / stage
        if not stage_dir.exists():
            raise ValueError(f"Stage directory does not exist: {stage_dir}")
        
        self.shard_files = sorted(list(stage_dir.glob("*.torch")))
        if len(self.shard_files) == 0:
            raise ValueError(f"No .torch files found in {stage_dir}")
        
        print(f"Found {len(self.shard_files)} shard files in {stage_dir}")
        
        # Create view sampler
        class SamplerCfg:
            def __init__(self, num_input_views, num_target_views):
                self.num_input_views = num_input_views
                self.num_target_views = num_target_views
        
        sampler_cfg = SamplerCfg(num_input_views, num_target_views)

        if view_sampler is None:
            self.view_sampler = ViewSamplerDefault(sampler_cfg, stage)
        else: 
            self.view_sampler = view_sampler(sampler_cfg, stage)

        self.current_step = 0  # tracking steps for pair distances
        self.dataset_name = "acid"
    
    def set_training_step(self, step: int):
        """Update current training step for baseline expansion."""
        self.current_step = step
    
    def decode_jpeg(self, jpeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decode 1D JPEG tensor to RGB image.
        Returns: RGB image tensor of shape [3, H, W], float32 in [0, 1]
        """
        jpeg_bytes = jpeg_tensor.cpu().numpy().tobytes()
        image = Image.open(BytesIO(jpeg_bytes)).convert('RGB')
        image = image.resize((self.target_image_size[1], self.target_image_size[0]), Image.LANCZOS)
        image_tensor = TF.to_tensor(image)
        return image_tensor
    
    def parse_cameras(self, cameras: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parse camera parameters into intrinsics and extrinsics.
        
        Args:
            cameras: [V, 18] tensor with format:
                [fx, fy, cx, cy, 0, 0, R00, R01, R02, t0, R10, R11, R12, t1, R20, R21, R22, t2]
                Note: fx, fy, cx, cy are normalized to [0, 1]
        
        Returns:
            intrinsics: [V, 3, 3] camera intrinsics (denormalized for target_image_size)
            extrinsics: [V, 4, 4] camera extrinsics (world-to-camera transform)
        """
        V = cameras.shape[0]
        
        # Extract intrinsics (normalized)
        fx_norm = cameras[:, 0]
        fy_norm = cameras[:, 1]
        cx_norm = cameras[:, 2]
        cy_norm = cameras[:, 3]
        
        # Denormalize intrinsics for actual image size
        fx = fx_norm * self.target_image_size[1]
        fy = fy_norm * self.target_image_size[0]
        cx = cx_norm * self.target_image_size[1]
        cy = cy_norm * self.target_image_size[0]
        
        # Build intrinsics matrix [V, 3, 3]
        intrinsics = torch.zeros(V, 3, 3, dtype=cameras.dtype, device=cameras.device)
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        intrinsics[:, 2, 2] = 1.0
        
        # Extract rotation and translation
        # cameras[:, 6:15] contains R (row-major: R00, R01, R02, R10, R11, R12, R20, R21, R22)
        # cameras[:, 9::4][:3] = t0, t1, t2 but let's be explicit
        R = cameras[:, 6:15].reshape(V, 3, 3)
        t = torch.stack([cameras[:, 9], cameras[:, 13], cameras[:, 17]], dim=1)  # [V, 3]
        
        # Build extrinsics [V, 4, 4] as [R | t; 0 0 0 1]
        extrinsics = torch.zeros(V, 4, 4, dtype=cameras.dtype, device=cameras.device)
        extrinsics[:, :3, :3] = R
        extrinsics[:, :3, 3] = t
        extrinsics[:, 3, 3] = 1.0
        
        return intrinsics, extrinsics
    
    def load_scene(self, scene_dict: dict) -> Optional[ViewSet]:
        """
        Load a scene from dictionary format into a ViewSet.
        """
        try:
            timestamps = scene_dict['timestamps']  # [V]
            cameras = scene_dict['cameras']  # [V, 18]
            jpeg_images = scene_dict['images']  # list of length V
            
            num_views = len(jpeg_images)
            
            # Decode all images
            images = []
            for jpeg_tensor in jpeg_images:
                img = self.decode_jpeg(jpeg_tensor)
                images.append(img)
            
            images = torch.stack(images, dim=0)
            
            # Parse cameras
            intrinsics, extrinsics = self.parse_cameras(cameras)
            
            viewset = ViewSet(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics
            )
            
            return viewset
            
        except Exception as e:
            print(f"Error loading scene {scene_dict.get('key', 'unknown')}: {e}")
            return None
    
    def __iter__(self):
        """Iterate over all scenes in all shards."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            shard_files = self.shard_files
        else:
            # Multi-process data loading: split shards across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_files = [f for i, f in enumerate(self.shard_files) if i % num_workers == worker_id]
        
        for shard_file in shard_files:
            try:
                # Load shard (list of scene dicts)
                scenes = torch.load(shard_file)
                
                for scene_dict in scenes:
                    # Load full scene as ViewSet
                    all_views = self.load_scene(scene_dict)
                    
                    if all_views is None:
                        continue
                    
                    # Sample context and target views
                    context_views, target_views = self.view_sampler.sample_views(
                        all_views,
                        curr_train_step=self.current_step if self.stage == 'train' else None,
                        max_train_steps=self.max_train_steps
                    )
                    
                    # Return as a batch-ready dict
                    batch = {
                        'context': {
                            'images': context_views.images,  # [num_input_views, 3, H, W]
                            'intrinsics': context_views.intrinsics,  # [num_input_views, 3, 3]
                            'extrinsics': context_views.extrinsics,  # [num_input_views, 4, 4]
                        },
                        'target': {
                            'images': target_views.images,  # [num_target_views, 3, H, W]
                            'intrinsics': target_views.intrinsics,  # [num_target_views, 3, 3]
                            'extrinsics': target_views.extrinsics,  # [num_target_views, 4, 4]
                        },
                        'scene_key': scene_dict.get('key', 'unknown'),
                        'near_plane': scene_dict.get('near_plane', 0.1),
                        'far_plane': scene_dict.get('far_plane', 100.0),
                    }
                    yield batch
                    
            except Exception as e:
                print(f"Error loading shard {shard_file}: {e}")
                continue
