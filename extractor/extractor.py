import torch
import lightning as L

from .cnn_extractor import CNNExtractor
from .transformer_extractor import TransformerExtractor
from utils import patchify


class Extractor(L.LightningModule):
    """
    Extractor module for the feature extractor.
    The layer order is: cnn_extractor->transformer_extractor
    The number of input channels is fixed to 3.
    """
    def __init__(
        self, image_size=256, hidden_dim=128, swin_divisions=2, 
        cnn_dtype=torch.float32, transformer_dtype=torch.float32
    ):
        """
        Initialize the Extractor.
        Args:
            image_size (int): size of the image
            hidden_dim (int): number of hidden dimensions
            swin_divisions (int): number of divisions for the swin transformer (total num of windows is swin_divisions**2)
            cnn_dtype (torch.dtype): data type for the cnn extractor
            transformer_dtype (torch.dtype): data type for the transformer extractor
        """
        super().__init__()
        self.dim = hidden_dim
        window_size = image_size // 4 // swin_divisions
        shift_size = window_size // 2
        self.cnn_extractor = CNNExtractor(hidden_dim, dtype=cnn_dtype)
        self.transformer_extractor = TransformerExtractor(
            hidden_dim, window_size, shift_size, dtype=transformer_dtype
        )

    def forward(self, x):
        """
        Forward pass of the Extractor.
        Args:
            x (torch.Tensor): input tensor of shape [B, K, 3, H, W]
        Returns:
            out (torch.Tensor): output tensor of shape [B, K, H//4, W//4, 128]
        """
        B, K, _, H, W = x.shape
        # assert torch.isfinite(x).all(), "input to extractor is not finite"
        x_cnn = self.cnn_extractor(x) # [B, K, 128, H//4, W//4]
        # assert torch.isfinite(x).all(), "activation after cnn extractor is not finite"
        x = patchify(x_cnn) # [B*K, K=(src|tgt), H//4, W//4, 128]
        # assert torch.isfinite(x).all(), "pachified data in extractor is not finite"
        x = x.to(self.transformer_extractor.dtype)
        x = self.transformer_extractor(x) # [B*K, H//4, W//4, 128]
        # assert torch.isfinite(x).all(), "output from transformer extractor is not finite"
        x = x.reshape(B, K, H//4, W//4, self.dim)
        return x, x_cnn # [B, K, H//4, W//4, 128], [B, K, 128, H//4, W//4]