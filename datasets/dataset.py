"""
Dataset configuration and initialization.
"""

from dataclasses import dataclass
from typing import Optional
from .dataset_acid import AcidDataset
from .dataset_re10k import Re10kDataset
from .dataset_mipnerf360 import MipNeRF360Dataset
from .dataset_tnt import TanksAndTemplesDataset
from .dataset_deepblending import DeepBlendingDataset
from .dataset_dtu import DTUDataset


@dataclass
class DatasetCfg:
    name: str  # ex. 'acid', 're10k', 'mipnerf360', 'tnt', 'deepblending'
    data_root: str  # path to dataset root directory
    stage: str = "train"  # 'train', 'test', or 'validation'
    num_input_views: int = 2
    num_target_views: int = 4
    target_image_size: int = 256  # target image size
    max_train_steps: int = 300000  # For baseline expansion


# Registry of available datasets
DATASETS = {
    "acid": AcidDataset,
    "re10k": Re10kDataset,
    "mipnerf360": MipNeRF360Dataset,
    "tnt": TanksAndTemplesDataset,
    "deepblending": DeepBlendingDataset,
    "dtu": DTUDataset,
}

def build_dataset(config: DatasetCfg):
    return DATASETS[config.name](
        data_root=config.data_root,
        stage=config.stage,
        num_input_views=config.num_input_views,
        num_target_views=config.num_target_views,
        target_image_size=config.target_image_size,
        max_train_steps=config.max_train_steps,
    )
