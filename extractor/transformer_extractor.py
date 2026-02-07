import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SwinCrossBlock


class TransformerExtractor(L.LightningModule):
    """
    Transformer extractor module for the feature extractor.
    The layers are: swin_block*6
    """
    # in [B*K, K=(src|tgt), H//4, W//4, 128]
    # out [B*K, H//4, W//4, 128]
    def __init__(self, dim=128, window_size=32, shift_size=16, num_heads=1, dtype=torch.float32):
        """
        Initialize the TransformerExtractor.
        Args:
            dim (int): number of hidden dimensions
            window_size (int): size of the window, basically the half of the image size (h or w)
            shift_size (int): size of the shift, basically the half of the window size
            num_heads (int): number of attention heads
            dtype (torch.dtype): data type
        """
        super().__init__()
        self.to(dtype)
        self.swin_block1 = SwinCrossBlock(dim, num_heads, window_size, shift_size=0, dtype=dtype)
        self.swin_block2 = SwinCrossBlock(dim, num_heads, window_size, shift_size=shift_size, dtype=dtype)
        self.swin_block3 = SwinCrossBlock(dim, num_heads, window_size, shift_size=0, dtype=dtype)
        self.swin_block4 = SwinCrossBlock(dim, num_heads, window_size, shift_size=shift_size, dtype=dtype)
        self.swin_block5 = SwinCrossBlock(dim, num_heads, window_size, shift_size=0, dtype=dtype)
        self.swin_block6 = SwinCrossBlock(dim, num_heads, window_size, shift_size=shift_size, dtype=dtype)

    def forward(self, x):
        """
        Forward pass of the TransformerExtractor.
        Args:
            x (torch.Tensor): input tensor of shape [B*K, K=(src|tgt), H//4, W//4, 128]
        Returns:
            out (torch.Tensor): output tensor of shape [B*K, H//4, W//4, 128]
        """
        # x: [B*K, K=(src|tgt), H//4, W//4, 128]
        x_src = x[:, 0, :, :, :] # [B*K, H//4, W//4, 128]
        x_tgt = x[:, 1:, :, :, :] # [B*K, K-1, H//4, W//4, 128]
        out = self.swin_block1(x_src, x_tgt)
        out = self.swin_block2(out, x_tgt)
        out = self.swin_block3(out, x_tgt)
        out = self.swin_block4(out, x_tgt)
        out = self.swin_block5(out, x_tgt)
        out = self.swin_block6(out, x_tgt)
        return out # [B*K, H//4, W//4, 128]
