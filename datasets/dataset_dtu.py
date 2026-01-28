"""
DTU Dataset

DTU MVS dataset with rectified images and camera parameters.

Directory structure:
    {data_root}/
        Rectified/
            scan{N}_train/
                rect_{camera_id}_{light_id}_r5000.png
        Cameras/
            {camera_id}_cam.txt  (contains extrinsic, intrinsic, near, far)
        Cameras/train/
            {camera_id}_cam.txt

Camera file format:
    extrinsic
    [4x4 world-to-camera matrix]
    
    intrinsic
    [3x3 intrinsic matrix with fx, fy, cx, cy]
    
    max_depth min_depth
"""

from dataclasses import dataclass
from torch.utils.data import IterableDataset
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from .view_sampler.view_sampler import ViewSet, ViewSampler, ViewSamplerDefault
from .shims.crop_shim import apply_crop_shim_to_views
from .shims.norm_shim import normalize_scene


@dataclass
class DTUDatasetCfg:
    data_root: str = "/workspace/re10kvol/dtu"
    stage: str = "train"  # 'train' or 'test'
    num_input_views: int = 2
    num_target_views: int = -1  # use ALL views if -1
    target_image_size: Tuple[int, int] = (256, 256)
    max_train_steps: int = 0
    view_sampler: ViewSampler = None
    light_idx: int = 3  # Which lighting condition to use (0-6)


class DTUDataset(IterableDataset):
    """
    DTU MVS dataset loader.
    
    Reads camera parameters from text files and rectified images.
    """
    
    def __init__(
        self,
        data_root: str = "/workspace/re10kvol/dtu",
        stage: str = "train",
        num_input_views: int = 2,
        num_target_views: int = -1,
        target_image_size: Tuple[int, int] = (256, 256),
        max_train_steps: int = 0,
        view_sampler: ViewSampler = None,
        light_idx: int = 3,
    ):
        """
        Args:
            data_root: Root directory containing Rectified/ and Cameras/ subdirectories
            stage: 'train' or 'test'
            num_input_views: Number of context/input views
            num_target_views: Number of target views (-1 = use all remaining views)
            target_image_size: Resize images to this size (H, W)
            max_train_steps: Maximum training steps (for baseline expansion schedule)
            view_sampler: Optional custom view sampler
            light_idx: Which lighting condition to use (0-6)
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.stage = stage
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        self.target_image_size = target_image_size
        self.max_train_steps = max_train_steps
        self.light_idx = light_idx
        self.dataset_name = "dtu"
        
        # Get scene list
        rectified_dir = self.data_root / "Rectified"
        if not rectified_dir.exists():
            raise ValueError(f"Rectified directory not found: {rectified_dir}")
        
        # Get all scan directories (e.g., scan1_train, scan2_train, ...)
        self.scene_dirs = sorted([d for d in rectified_dir.iterdir() 
                                  if d.is_dir() and d.name.endswith(f"_{stage}")])
        
        if len(self.scene_dirs) == 0:
            raise ValueError(f"No scenes found in {rectified_dir} for stage {stage}")
        
        print(f"Found {len(self.scene_dirs)} DTU scenes for {stage}")
        
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
    
    def set_training_step(self, step: int):
        """Update current training step for baseline expansion."""
        self.current_step = step
    
    def read_cam_file(self, cam_file: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Read DTU camera file.
        
        Returns:
            extrinsic: 4x4 world-to-camera matrix
            intrinsic: 3x3 intrinsic matrix
            near: near plane
            far: far plane
        """
        with open(cam_file, 'r') as f:
            lines = f.readlines()
        
        # Parse extrinsic (4x4 matrix)
        extrinsic_start = lines.index("extrinsic\n") + 1
        extrinsic = []
        for i in range(4):
            row = [float(x) for x in lines[extrinsic_start + i].strip().split()]
            extrinsic.append(row)
        extrinsic = np.array(extrinsic, dtype=np.float32)
        
        # Parse intrinsic (3x3 matrix)
        intrinsic_start = lines.index("intrinsic\n") + 1
        intrinsic = []
        for i in range(3):
            row = [float(x) for x in lines[intrinsic_start + i].strip().split()]
            intrinsic.append(row)
        intrinsic = np.array(intrinsic, dtype=np.float32)
        
        # Parse near and far planes (last line: max_depth min_depth)
        depth_line = lines[-1].strip().split()
        far = float(depth_line[0])  # max_depth
        near = float(depth_line[1])  # min_depth
        
        return extrinsic, intrinsic, near, far
    
    def get_camera_dir(self) -> Path:
        """Return path to camera directory based on stage."""
        if self.stage == "train":
            cam_dir = self.data_root / "Cameras" / "train"
            if cam_dir.exists():
                return cam_dir
        # Fallback to main Cameras directory
        return self.data_root / "Cameras"
    
    def load_scene(self, scene_dir: Path) -> Optional[ViewSet]:
        """
        Load a DTU scene into a ViewSet.
        """
        try:
            scene_name = scene_dir.name
            
            # Get all images with the specified lighting condition
            # Format: rect_{camera_id}_{light_id}_r5000.png
            image_files = sorted(scene_dir.glob(f"rect_*_{self.light_idx}_r5000.png"))
            
            if len(image_files) == 0:
                print(f"No images found for light {self.light_idx} in {scene_dir}")
                return None
            
            images = []
            intrinsics_list = []
            extrinsics_list = []
            near_planes = []
            far_planes = []
            
            cam_dir = self.get_camera_dir()
            
            for img_file in image_files:
                # Extract camera ID from filename (e.g., rect_001_3_r5000.png -> 001)
                filename = img_file.stem  # rect_001_3_r5000
                parts = filename.split('_')
                camera_id = parts[1]  # '001'
                
                # Read camera file
                cam_file = cam_dir / f"{str((int(camera_id) - 1)).zfill(8)}_cam.txt"
                if not cam_file.exists():
                    print(f"Camera file not found: {cam_file}")
                    continue
                
                # load image to tensor
                img = Image.open(img_file).convert('RGB')
                img_tensor = TF.to_tensor(img)  # [3, H, W]
                images.append(img_tensor)
                
                extrinsic, intrinsic, near, far = self.read_cam_file(cam_file)
                extrinsics_list.append(torch.from_numpy(extrinsic).float())
                intrinsics_list.append(torch.from_numpy(intrinsic).float())
                near_planes.append(near)
                far_planes.append(far)
            
            if len(images) == 0:
                return None
            
            # Stack all views
            images_tensor = torch.stack(images, dim=0)  # [V, 3, H, W]
            intrinsics_tensor = torch.stack(intrinsics_list, dim=0)  # [V, 3, 3]
            extrinsics_tensor = torch.stack(extrinsics_list, dim=0)  # [V, 4, 4]
            
            # Use min/max near/far planes for the scene
            min_near = np.min(near_planes)
            max_far = np.max(far_planes)
            
            viewset = ViewSet(
                extrinsics=extrinsics_tensor,
                intrinsics=intrinsics_tensor,
                images=images_tensor
            )
            
            return viewset, min_near, max_far
            
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
            result = self.load_scene(scene_dir)
            
            if result is None:
                continue
            
            all_views, min_near, max_far = result

            # Center and normalize scene using all cameras
            all_views, _ = normalize_scene(all_views)
            
            all_views = apply_crop_shim_to_views(all_views, self.target_image_size)
            
            # Sample context and target views
            context_views, target_views = self.view_sampler.sample_views(
                all_views,
                curr_train_step=self.current_step if self.stage == 'train' else None,
                max_train_steps=self.max_train_steps
            )
            
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
                'near_plane': min_near,
                'far_plane': max_far,
            }
            yield batch
