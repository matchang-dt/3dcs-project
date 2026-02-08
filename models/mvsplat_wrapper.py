import lightning as L
import torch

class MVSPlatWrapper(L.LightningModule):
    """
    MVSPlatWrapper module.
    Wraps the MVSPlat model for the depth estimation.
    """
    def __init__(self, dtype=torch.float32):
        """
        Initialize the MVSPlatWrapper.
        Args:
            dtype (torch.dtype): data type
        """
        super().__init__()
        pass

    def forward(self, x):
        pass