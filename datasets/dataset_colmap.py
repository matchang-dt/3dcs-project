"""
Base COLMAP Dataset

Common functionality for datasets using COLMAP sparse reconstruction format.
This includes reading COLMAP binary (.bin) and text (.txt) files and converting to ViewSet format.

Supports both COLMAP formats:
- Binary format: cameras.bin, images.bin (preferred if both exist)
- Text format: cameras.txt, images.txt (fallback)

Datasets that use COLMAP format:
- MipNeRF360
- Tanks & Temples
- Deep Blending
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
from .shims.crop_shim import apply_crop_shim_to_views, update_intrinsics_for_resize, update_intrinsics_for_crop
from .shims.norm_shim import normalize_scene


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
            
            # Determine number of parameters based on model
            # 0: SIMPLE_PINHOLE, 1: PINHOLE, 2: SIMPLE_RADIAL, 3: RADIAL
            if model_id == 0: num_params = 3
            elif model_id == 1: num_params = 4
            elif model_id == 2: num_params = 4
            elif model_id == 3: num_params = 5
            else: num_params = 4 # Default fallback
            
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            
            # Standardize to (fx, fy, cx, cy)
            if model_id == 0 or model_id == 2: # SIMPLE_PINHOLE or SIMPLE_RADIAL
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else: # PINHOLE and others
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]

            cameras[camera_id] = {
                'id': camera_id,
                'model': model_id,
                'width': width,
                'height': height,
                'params': (fx, fy, cx, cy)
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


def read_cameras_text(path_to_model_file):
    """
    Read COLMAP cameras.txt file.
    Returns dict of Camera objects with camera_id as key.
    
    Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    For PINHOLE: CAMERA_ID, PINHOLE, WIDTH, HEIGHT, FX, FY, CX, CY
    """
    cameras = {}
    with open(path_to_model_file, "r") as fid:
        for line in fid:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            
            # Parse parameters (for PINHOLE: fx, fy, cx, cy)
            params = tuple(float(p) for p in parts[4:])
            
            cameras[camera_id] = {
                'id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_images_text(path_to_model_file):
    """
    Read COLMAP images.txt file.
    Returns dict of Image objects with image_id as key.
    
    Format: 
    - Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    - Line 2: 2D points (X Y POINT3D_ID X Y POINT3D_ID ...) - we skip this
    - Lines alternate: image data, then 2D points, repeat
    """
    images = {}
    with open(path_to_model_file, "r") as fid:
        lines = fid.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                i += 1
                continue
            
            parts = line.split()
            # Image lines have at least 10 parts (IMAGE_ID + 8 pose params + CAMERA_ID + NAME)
            # 2D points lines have variable length (multiple of 3: X Y POINT3D_ID)
            if len(parts) >= 10:
                # This is an image line
                try:
                    image_id = int(parts[0])
                    qw = float(parts[1])
                    qx = float(parts[2])
                    qy = float(parts[3])
                    qz = float(parts[4])
                    tx = float(parts[5])
                    ty = float(parts[6])
                    tz = float(parts[7])
                    camera_id = int(parts[8])
                    # Image name might have spaces, so join the rest
                    image_name = " ".join(parts[9:])
                    
                    qvec = np.array([qw, qx, qy, qz])
                    tvec = np.array([tx, ty, tz])
                    
                    images[image_id] = {
                        'id': image_id,
                        'qvec': qvec,
                        'tvec': tvec,
                        'camera_id': camera_id,
                        'name': image_name,
                    }
                except (ValueError, IndexError):
                    # Skip malformed lines
                    pass
                # Skip the next line (2D points) and move to next image
                i += 2
            else:
                # This might be a 2D points line or empty, skip it
                i += 1
    
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
        Supports both binary (.bin) and text (.txt) COLMAP formats.
        """
        try:
            # Read COLMAP data - check for both binary and text formats
            colmap_dir = self.get_colmap_dir(scene_dir)
            cameras_bin = colmap_dir / "cameras.bin"
            images_bin = colmap_dir / "images.bin"
            cameras_txt = colmap_dir / "cameras.txt"
            images_txt = colmap_dir / "images.txt"
            
            # Determine format (prefer binary if both exist)
            use_binary = cameras_bin.exists() and images_bin.exists()
            use_text = cameras_txt.exists() and images_txt.exists()
            
            if not use_binary and not use_text:
                print(f"Missing COLMAP files in {colmap_dir} (checked for .bin and .txt)")
                return None
            
            if use_binary:
                cameras = read_cameras_binary(str(cameras_bin))
                colmap_images = read_images_binary(str(images_bin))
            else:
                cameras = read_cameras_text(str(cameras_txt))
                colmap_images = read_images_text(str(images_txt))
            
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
                W_actual, H_actual = img.size
                
                # Get camera intrinsics
                camera = cameras[img_info['camera_id']]
                fx, fy, cx, cy = camera['params']
                W_expected = camera['width']
                H_expected = camera['height']
                
                # AUTO-SCALE DETECTION
                # If images on disk are already downsampled (offline), we adjust intrinsics automatically
                auto_scale_x = W_actual / W_expected
                auto_scale_y = H_actual / H_expected

                # Initialize intrinsics matrix (with batch dim for helper functions)
                K = torch.eye(3, dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]
                K[:, 0, 0] = fx
                K[:, 1, 1] = fy
                K[:, 0, 2] = cx
                K[:, 1, 2] = cy
                
                # Step 1: Apply auto-scale if images are pre-downsampled
                if abs(auto_scale_x - 1.0) > 0.001 or abs(auto_scale_y - 1.0) > 0.001:
                    K = update_intrinsics_for_resize(K, (H_expected, W_expected), (H_actual, W_actual))
                
                # Step 2: Center crop to square (maintain aspect ratio)
                crop_size = min(H_actual, W_actual)
                left = (W_actual - crop_size) // 2
                top = (H_actual - crop_size) // 2
                img = img.crop((left, top, left + crop_size, top + crop_size))
                W_cropped, H_cropped = crop_size, crop_size
                
                # Adjust intrinsics for center crop
                K = update_intrinsics_for_crop(K, (H_actual, W_actual), (H_cropped, W_cropped))
                
                # Step 3: Resize to target_image_size
                H_target, W_target = self.target_image_size if isinstance(self.target_image_size, tuple) else (self.target_image_size, self.target_image_size)
                img = img.resize((W_target, H_target), Image.LANCZOS)
                
                # Adjust intrinsics for resize
                K = update_intrinsics_for_resize(K, (H_cropped, W_cropped), (H_target, W_target))
                
                # Remove batch dimension
                K = K.squeeze(0)  # [3, 3]
                img_tensor = TF.to_tensor(img)  # [3, H, W]
                images.append(img_tensor)
                intrinsics_list.append(K)
                
                # Get camera extrinsics (COLMAP is w2c)
                R_w2c = qvec2rotmat(img_info['qvec'])
                t_w2c = img_info['tvec']
                
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
                extrinsics=extrinsics_tensor,
                intrinsics=intrinsics_tensor,
                images=images_tensor
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
            
            # Center and normalize scene using all cameras
            all_views, _ = normalize_scene(all_views)
            
            if all_views is None:
                continue
            
            # Note: Images are already center-cropped and resized to target_image_size in load_scene()
            # The crop_shim is not needed here since processing is done during loading
            
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
