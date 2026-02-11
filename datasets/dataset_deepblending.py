"""
Deep Blending Dataset

Loads scenes from Deep Blending format with COLMAP sparse reconstruction.
Directory structure (after reorganization):
    {data_root}/{scene_name}/
        images/         # Images
        sparse/         # COLMAP reconstruction (or sparse/0/)
            cameras.bin
            images.bin
            points3D.bin

Deep Blending scenes:
- Aquarium-20, Bedroom, Boats, Bridge, CreepyAttic, DrJohnson, Hugo-1
- Library, Lumber, Museum-1, Museum-2, NightSnow, Playroom, Ponche
- SaintAnne, Shed, Street-10, Tree-18, Yellowhouse-12
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

from .dataset_colmap import ColmapDataset
from .view_sampler.view_sampler import ViewSampler


@dataclass
class DeepBlendingDatasetCfg:
    data_root: str = "/workspace/re10kvol/deepblending"
    stage: str = "test"  # eval only dataset
    num_input_views: int = 2
    num_target_views: int = -1  # use ALL views if -1
    target_image_size: Tuple[int, int] = (256, 256)
    max_train_steps: Optional[int] = None
    view_sampler: ViewSampler = None


class DeepBlendingDataset(ColmapDataset):
    """
    Deep Blending dataset loader using COLMAP base class.
    
    Note: After reorganization, the structure should be:
    deepblending/
      {scene_name}/
        images/
        sparse/  (or sparse/0/)
    """
    
    def __init__(
        self,
        data_root: str = "/workspace/re10kvol/deepblending",
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
            dataset_name="deepblending",
        )
    
    def get_scene_list(self) -> List[Path]:
        """Return list of Deep Blending scene directories."""
        return [d for d in self.data_root.iterdir() if d.is_dir()]
    
    def get_image_dir(self, scene_dir: Path) -> Path:
        """Return path to image directory for a given scene."""
        return scene_dir / "images"
    
    def get_colmap_dir(self, scene_dir: Path) -> Path:
        """Return path to COLMAP sparse reconstruction directory."""
        # Check for sparse/0/ first (like MipNeRF360), then sparse/ (like TNT)
        if (scene_dir / "sparse" / "0").exists():
            return scene_dir / "sparse" / "0"
        return scene_dir / "sparse"
