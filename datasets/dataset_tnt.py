"""
Tanks & Temples Dataset

Loads scenes from Tanks & Temples format with COLMAP sparse reconstruction.
Directory structure:
    {data_root}/{scene_name}/
        images/         # Images
        sparse/         # COLMAP reconstruction
            cameras.bin
            images.bin
            points3D.bin

Common Tanks & Temples scenes:
- Barn, Caterpillar, Church, Courthouse, Ignatius, Meetingroom, Truck
- Family, Francis, Horse, Lighthouse, M60, Panther, Playground, Train
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path

from .dataset_colmap import ColmapDataset
from .view_sampler.view_sampler import ViewSampler


@dataclass
class TanksAndTemplesDatasetCfg:
    data_root: str = "/workspace/re10kvol/tnt"
    stage: str = "test"  # eval only dataset
    num_input_views: int = 2
    num_target_views: int = -1  # use ALL views if -1
    target_image_size: int = 256
    max_train_steps: int = 0
    view_sampler: ViewSampler = None


class TanksAndTemplesDataset(ColmapDataset):
    """
    Tanks & Temples dataset loader using COLMAP base class.
    
    Note: Tanks & Temples uses a slightly different directory structure than MipNeRF360,
    but the COLMAP format is the same.
    """
    
    def __init__(
        self,
        data_root: str = "/workspace/re10kvol/tnt",
        stage: str = "test",
        num_input_views: int = 2,
        num_target_views: int = -1,
        target_image_size: int = 256,
        max_train_steps: int = 0,
        view_sampler: ViewSampler = None,
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
        """
        super().__init__(
            data_root=data_root,
            stage=stage,
            num_input_views=num_input_views,
            num_target_views=num_target_views,
            target_image_size=target_image_size,
            max_train_steps=max_train_steps,
            view_sampler=view_sampler,
            dataset_name="tanks_and_temples",
        )
    
    def get_scene_list(self) -> List[Path]:
        """Return list of Tanks & Temples scene directories."""
        return [d for d in self.data_root.iterdir() if d.is_dir()]
    
    def get_image_dir(self, scene_dir: Path) -> Path:
        """Return path to image directory for a given scene."""
        return scene_dir / "images"
    
    def get_colmap_dir(self, scene_dir: Path) -> Path:
        """Return path to COLMAP sparse reconstruction directory."""
        # Tanks & Temples typically has sparse/ directly (not sparse/0/)
        # But check for sparse/0/ first for compatibility
        if (scene_dir / "sparse" / "0").exists():
            return scene_dir / "sparse" / "0"
        return scene_dir / "sparse"
