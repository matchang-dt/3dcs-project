import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SwinCrossBlock


class TransformerExtractor(L.LightningModule):
    # in [B*K, K=(src|tgt), H//4, W//4, 128]
    # out [B*K, H//4, W//4, 128]
    def __init__(self, dim=128, window_size=32, shift_size=16, num_heads=1, dtype=torch.float32):
        super().__init__()
        self.to(dtype)
        self.swin_block1 = SwinCrossBlock(dim, num_heads, window_size, shift_size=0, dtype=dtype)
        self.swin_block2 = SwinCrossBlock(dim, num_heads, window_size, shift_size=shift_size, dtype=dtype)
        self.swin_block3 = SwinCrossBlock(dim, num_heads, window_size, shift_size=0, dtype=dtype)
        self.swin_block4 = SwinCrossBlock(dim, num_heads, window_size, shift_size=shift_size, dtype=dtype)
        self.swin_block5 = SwinCrossBlock(dim, num_heads, window_size, shift_size=0, dtype=dtype)
        self.swin_block6 = SwinCrossBlock(dim, num_heads, window_size, shift_size=shift_size, dtype=dtype)

    def forward(self, x):
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
