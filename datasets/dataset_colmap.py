"""
Base COLMAP Dataset

Common functionality for datasets using COLMAP sparse reconstruction format.
This includes reading COLMAP binary files and converting to ViewSet format.

Datasets that use COLMAP format:
- MipNeRF360
- Tanks & Temples
- DTU (some variants)
- Custom COLMAP reconstructions
"""

import os
import struct
import torch
import torchvision.transforms.functional as TF
from abc import ABC, abstractmethod
from torch.utils.data import IterableDataset
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

from .view_sampler.view_sampler import ViewSet, ViewSampler, ViewSamplerDefault


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP cameras.bin file.
    Returns dict of Camera objects with camera_id as key.
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 4  # Assuming PINHOLE model (fx, fy, cx, cy)
            
            # Read camera parameters
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            
            cameras[camera_id] = {
                'id': camera_id,
                'model': model_id,
                'width': width,
                'height': height,
                'params': params  # (fx, fy, cx, cy) for PINHOLE
            }
    return cameras


def read_images_binary(path_to_model_file):
    """
    Read COLMAP images.bin file.
    Returns dict of Image objects with image_id as key.
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qw = binary_image_properties[1]
            qx = binary_image_properties[2]
            qy = binary_image_properties[3]
            qz = binary_image_properties[4]
            tx = binary_image_properties[5]
            ty = binary_image_properties[6]
            tz = binary_image_properties[7]
            camera_id = binary_image_properties[8]
            
            # Read image name
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # Read 2D points (we skip these for now)
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            
            # Convert quaternion to rotation matrix
            qvec = np.array([qw, qx, qy, qz])
            tvec = np.array([tx, ty, tz])
            
            images[image_id] = {
                'id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
            }
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


class ColmapDataset(IterableDataset, ABC):
    """
    Base class for datasets using COLMAP sparse reconstruction format.
    
    Subclasses should implement:
    - get_scene_list(): Return list of scene directories
    - get_image_dir(scene_dir): Return path to image directory for a scene
    - get_colmap_dir(scene_dir): Return path to COLMAP sparse directory
    """
    
    def __init__(
        self,
        data_root: str,
        stage: str = "test",
        num_input_views: int = 2,
        num_target_views: int = -1,
        target_image_size: int = 256,
        max_train_steps: int = 0,
        view_sampler: ViewSampler = None,
        dataset_name: str = "colmap",
    ):
        """
        Args:
            data_root: Root directory containing scene subdirectories
            stage: 'train' or 'test'
            num_input_views: Number of context/input views
            num_target_views: Number of target views (-1 = use all remaining views)
            target_image_size: Resize images to this size (height=width)
            max_train_steps: Maximum training steps (for baseline expansion schedule)
            view_sampler: Optional custom view sampler
            dataset_name: Name of the dataset (for identification)
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.stage = stage
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.target_image_size = target_image_size
        self.max_train_steps = max_train_steps
        self.dataset_name = dataset_name
        
        # Get scene list from subclass
        self.scene_dirs = self.get_scene_list()
        if len(self.scene_dirs) == 0:
            raise ValueError(f"No scenes found in {self.data_root}")
        
        print(f"Found {len(self.scene_dirs)} scenes in {self.data_root}")
        
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
        
        self.current_step = 0
    
    @abstractmethod
    def get_scene_list(self) -> List[Path]:
        """Return list of scene directories to process."""
        pass
    
    @abstractmethod
    def get_image_dir(self, scene_dir: Path) -> Path:
        """Return path to image directory for a given scene."""
        pass
    
    @abstractmethod
    def get_colmap_dir(self, scene_dir: Path) -> Path:
        """Return path to COLMAP sparse reconstruction directory."""
        pass
    
    def set_training_step(self, step: int):
        """Update current training step for baseline expansion."""
        self.current_step = step
    
    def load_scene(self, scene_dir: Path) -> Optional[ViewSet]:
        """
        Load a scene from COLMAP format into a ViewSet.
        """
        try:
            # Read COLMAP data
            colmap_dir = self.get_colmap_dir(scene_dir)
            cameras_file = colmap_dir / "cameras.bin"
            images_file = colmap_dir / "images.bin"
            
            if not cameras_file.exists() or not images_file.exists():
                print(f"Missing COLMAP files in {colmap_dir}")
                return None
            
            cameras = read_cameras_binary(str(cameras_file))
            colmap_images = read_images_binary(str(images_file))
            
            # Sort images by name for consistency
            image_list = sorted(colmap_images.values(), key=lambda x: x['name'])
            
            images = []
            intrinsics_list = []
            extrinsics_list = []
            
            image_dir = self.get_image_dir(scene_dir)
            
            for img_info in image_list:
                # Load image
                img_path = image_dir / img_info['name']
                if not img_path.exists():
                    print(f"Image not found: {img_path}")
                    continue
                
                img = Image.open(img_path).convert('RGB')
                # Resize to target size (handle both int and tuple)
                if isinstance(self.target_image_size, tuple):
                    target_size = self.target_image_size
                else:
                    target_size = (self.target_image_size, self.target_image_size)
                img = img.resize(target_size, Image.LANCZOS)
                img_tensor = TF.to_tensor(img)  # [3, H, W]
                images.append(img_tensor)
                
                # Get camera intrinsics
                camera = cameras[img_info['camera_id']]
                fx, fy, cx, cy = camera['params']
                orig_width = camera['width']
                orig_height = camera['height']
                
                # Scale intrinsics to target image size (handle both int and tuple)
                if isinstance(self.target_image_size, tuple):
                    target_h, target_w = self.target_image_size
                else:
                    target_h = target_w = self.target_image_size
                scale_x = target_w / orig_width
                scale_y = target_h / orig_height
                
                fx_scaled = fx * scale_x
                fy_scaled = fy * scale_y
                cx_scaled = cx * scale_x
                cy_scaled = cy * scale_y
                
                K = torch.eye(3, dtype=torch.float32)
                K[0, 0] = fx_scaled
                K[1, 1] = fy_scaled
                K[0, 2] = cx_scaled
                K[1, 2] = cy_scaled
                intrinsics_list.append(K)
                
                # Get camera extrinsics (world-to-camera)
                # COLMAP stores camera-to-world, so we need to invert
                R_c2w = qvec2rotmat(img_info['qvec'])
                t_c2w = img_info['tvec']
                
                # Convert to world-to-camera
                R_w2c = R_c2w.T
                t_w2c = -R_w2c @ t_c2w
                
                extrinsics = torch.eye(4, dtype=torch.float32)
                extrinsics[:3, :3] = torch.from_numpy(R_w2c).float()
                extrinsics[:3, 3] = torch.from_numpy(t_w2c).float()
                extrinsics_list.append(extrinsics)
            
            if len(images) == 0:
                return None
            
            # Stack all views
            images_tensor = torch.stack(images, dim=0)  # [V, 3, H, W]
            intrinsics_tensor = torch.stack(intrinsics_list, dim=0)  # [V, 3, 3]
            extrinsics_tensor = torch.stack(extrinsics_list, dim=0)  # [V, 4, 4]
            
            viewset = ViewSet(
                images=images_tensor,
                intrinsics=intrinsics_tensor,
                extrinsics=extrinsics_tensor
            )
            
            return viewset
            
        except Exception as e:
            print(f"Error loading scene {scene_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def __iter__(self):
        """Iterate over all scenes."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            scene_dirs = self.scene_dirs
        else:
            # Multi-process data loading: split scenes across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            scene_dirs = [d for i, d in enumerate(self.scene_dirs) if i % num_workers == worker_id]
        
        for scene_dir in scene_dirs:
            # Load full scene as ViewSet
            all_views = self.load_scene(scene_dir)
            
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
                'scene_key': scene_dir.name,
                'near_plane': 0.1,  # Default values, can be overridden by subclasses
                'far_plane': 100.0,
            }
            yield batch
