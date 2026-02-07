import torch
import lightning as L

from .cnn_extractor import CNNExtractor
from .transformer_extractor import TransformerExtractor


def patchify(x: torch.Tensor) -> torch.Tensor:
    b, k, c, h, w = x.shape
    x = x.permute(0, 1, 3, 4, 2) # [B, K, 128, H//4, W//4] -> [B, K, H//4, W//4, 128]
    patches = []
    for i in range(b):
        for j in range(k):
            src = x[i, j, :, :, :].unsqueeze(0) # [1, H//4, W//4, 128]
            tgt1 = x[i, :j, :, :, :] 
            tgt2 = x[i, j + 1:, :, :, :]
            tgt = torch.cat([tgt1, tgt2], dim=0) # [K - 1, H//4, W//4, 128]
            ex = torch.cat([src, tgt], dim=0) # [K=(src|tgt), H//4, W//4, 128]
            patches.append(ex)
    patches = torch.stack(patches, dim=0) # [B * K, K=(src|tgt), H//4, W//4, 128]
    return patches


class Extractor(L.LightningModule):
    def __init__(
        self, image_size=256, hidden_dim=128, swin_divisions=2, 
        cnn_dtype=torch.float32, transformer_dtype=torch.float32
    ):
        super().__init__()
        self.dim = hidden_dim
        window_size = image_size // 4 // swin_divisions
        shift_size = window_size // 2
        self.cnn_extractor = CNNExtractor(hidden_dim, dtype=cnn_dtype)
        self.transformer_extractor = TransformerExtractor(
            hidden_dim, window_size, shift_size, dtype=transformer_dtype
        )

    def forward(self, x):
        B, K, _, H, W = x.shape
        x = self.cnn_extractor(x) # [B, K, 128, H//4, W//4]
        x = patchify(x) # [B*K, K=(src|tgt), H//4, W//4, 128]
        x = x.to(self.transformer_extractor.dtype)
        x = self.transformer_extractor(x) # [B*K, H//4, W//4, 128]
        x = x.reshape(B, K, H//4, W//4, self.dim)
        return x # [B, K, H//4, W//4, 128]