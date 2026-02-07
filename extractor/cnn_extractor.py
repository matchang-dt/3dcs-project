import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ResBlock4Extractor

        
class CNNExtractor(L.LightningModule): # [B, K, 3, H, W] -> [B, K, 128, H//4, W//4]
    """
    CNN extractor module for the feature extractor.
    The layer order is: conv1->bn1->silu->resblocks*6->proj
    The number of input channels is fixed to 3.
    """
    def __init__(self, out_channels=128, dtype=torch.float32):
        """
        Initialize the CNNExtractor.
        Args:
            out_channels (int): number of output channels
            dtype (torch.dtype): data type
        """
        super().__init__()
        self.to(dtype)
        hidden_channels1 = out_channels // 4
        hidden_channels2 = out_channels // 2
        self.conv1 = nn.Conv2d(3, hidden_channels1, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(hidden_channels1, dtype=dtype)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.silu = nn.SiLU()
        self.res1 = ResBlock4Extractor(hidden_channels1, hidden_channels1, dtype=dtype)
        self.res2 = ResBlock4Extractor(hidden_channels1, hidden_channels1, dtype=dtype)
        self.res3 = ResBlock4Extractor(hidden_channels1, hidden_channels1, dtype=dtype)
        self.res4 = ResBlock4Extractor(hidden_channels1, hidden_channels2, stride=2, dtype=dtype)
        self.res5 = ResBlock4Extractor(hidden_channels2, hidden_channels2, dtype=dtype)
        self.res6 = ResBlock4Extractor(hidden_channels2, out_channels, stride=2, dtype=dtype)
        self.proj = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=1, stride=1, padding=0, bias=True, dtype=dtype
        )
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        """
        Forward pass of the CNNExtractor.
        Args:
            x (torch.Tensor): input tensor of shape [B, K, 3, H, W]
        Returns:
            out (torch.Tensor): output tensor of shape [B, K, 128, H//4, W//4]
        """
        # x: [B, K, 3, H, W] 
        if x.dim() == 4:
            b = 1
            k = x.shape[0]
        else:
            b, k = x.shape[:2]
            x = x.view(b * k, *x.shape[-3:]) # [B * K, 3, H, W]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.proj(out)
        return out.reshape(b, k, *out.shape[-3:]) # [B, K, 128, H//4, W//4]


if __name__ == '__main__':
    x = torch.randn(3, 4, 3, 32, 32)
    extractor = CNNExtractor()
    out = extractor(x)
    print(out.shape)