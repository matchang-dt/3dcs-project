"""
MipNeRF360 Dataset

Loads scenes from MipNeRF360 format with COLMAP sparse reconstruction.
Directory structure:
    {data_root}/{scene_name}/
        images/         # Full resolution images
        images_2/       # 1/2 resolution
        images_4/       # 1/4 resolution
        images_8/       # 1/8 resolution
        sparse/0/       # COLMAP reconstruction
            cameras.bin
            images.bin
            points3D.bin

Outputs same format as ACID dataset for compatibility.
"""

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

from .dataset_colmap import ColmapDataset
from .view_sampler.view_sampler import ViewSampler


@dataclass
class MipNeRF360DatasetCfg:
    data_root: str = "/workspace/re10kvol/mipnerf360"
    stage: str = "test"  # eval only dataset
    num_input_views: int = 2
    num_target_views: int = -1  # use ALL views if -1
    target_image_size: Tuple[int, int] = (256, 256)
    max_train_steps: int = 300000
    view_sampler: ViewSampler = None


class MipNeRF360Dataset(ColmapDataset):
    """
    MipNeRF360 dataset loader using COLMAP base class.
    
    MipNeRF360 scenes:
    - bicycle, bonsai, counter, garden, kitchen, room, stump
    """
    
    def __init__(
        self,
        data_root: str = "/workspace/re10kvol/mipnerf360",
        stage: str = "test",
        num_input_views: int = 2,
        num_target_views: int = -1,
        target_image_size: Tuple[int, int] = (256, 256),
        max_train_steps: int = 0,
        view_sampler: ViewSampler = None,
    ):
        """
        Args:
            data_root: Root directory containing scene subdirectories
            stage: 'train' or 'test' (MipNeRF360 doesn't have predefined splits)
            num_input_views: Number of context/input views
            num_target_views: Number of target views (-1 = use all remaining views)
            target_image_size: Resize images to this size (height=width)
            max_train_steps: Maximum training steps (for baseline expansion schedule)
            view_sampler: Optional custom view sampler
        """
        super().__init__(
            data_root=data_root,
            stage=stage,
            num_input_views=num_input_views,
            num_target_views=num_target_views,
            target_image_size=target_image_size,
            max_train_steps=max_train_steps,
            view_sampler=view_sampler,
            dataset_name="mipnerf360",
        )
    
    def get_scene_list(self) -> List[Path]:
        """Return list of MipNeRF360 scene directories."""
        return [d for d in self.data_root.iterdir() if d.is_dir()]
    
    def get_image_dir(self, scene_dir: Path) -> Path:
        """Return path to image directory for a given scene."""
        return scene_dir / "images"
    
    def get_colmap_dir(self, scene_dir: Path) -> Path:
        """Return path to COLMAP sparse reconstruction directory."""
        return scene_dir / "sparse" / "0"
