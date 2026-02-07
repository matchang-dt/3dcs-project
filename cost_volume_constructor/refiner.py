import lightning as L
import torch
from torch import nn

from utils import WindowSelfAttention, WindowCrossAttention, ResBlock4UNet, patchify


class CostVolumeRefiner(L.LightningModule):
    """
    Cost volume refiner module.
    Refines the cost volume with the features and the images by a U-net based refiner.
    """
    def __init__(self, channels=128, feat_map_size=64, dtype=torch.float32):
        """
        Initialize the CostVolumeRefiner.
        Args:
            channels (int): number of channels for the features
            feat_map_size (int): size of the feature map
            dtype (torch.dtype): data type
        """
        super().__init__()
        self.channels = channels
        self.to(dtype)
        self.res_enc1_1 = ResBlock4UNet(channels*2, channels, dtype)
        self.res_enc1_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_enc2_1 = ResBlock4UNet(channels, channels, dtype)
        self.res_enc2_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv2 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.up_conv1 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec1_1 = ResBlock4UNet(channels*2, channels, dtype)
        self.res_dec1_2 = ResBlock4UNet(channels, channels, dtype)
        self.up_conv2 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec2_1 = ResBlock4UNet(channels*2, channels, dtype)
        self.res_dec2_2 = ResBlock4UNet(channels, channels, dtype)
        self.self_attn = WindowSelfAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn1 = WindowCrossAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn2 = WindowCrossAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn3 = WindowCrossAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 window: not swinT
        nn.init.kaiming_normal_(self.down_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.down_conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_conv2.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        """
        Forward pass of the CostVolumeRefiner.
        Args:
            x (torch.Tensor): input tensor of shape [B, K, H//4, W//4, d (128*2)] concat of upsampled features (128) and cost volume (128)
        Returns:
            out (torch.Tensor): output tensor of shape [B, K, H//4, W//4, 128]
        """
        # x: [B, K, H//4, W//4, d (128*2)]
        b, k, h, w, d = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b*k, d, h, w) # [B, K, H//4, W//4, d] -> [B*K, d, H//4, W//4]
        # encoder
        h1 = self.res_enc1_1(x)
        h1 = self.res_enc1_2(h1)
        h2 = self.down_conv1(h1)
        h2 = self.res_enc2_1(h2)
        h2 = self.res_enc2_2(h2)
        h3 = self.down_conv2(h2) # [B*K, 128, H//16, W//16]

        # bottleneck
        h3 = patchify(h3.reshape(b, k, self.channels, h//4, w//4)) # [B*K, K=(src|tgt), H//16, W//16, 128]
        h_src = h3[:, 0, :, :, :].reshape(b*k, h//4 * w//4, self.channels) # [B*K, H//16, W//16, 128]
        h_tgt = h3[:, 1:, :, :, :].reshape(b*k, (k-1) * h//4 * w//4, self.channels) # [B*K, K-1, H//16, W//16, 128]
        h_src = self.self_attn(h_src)
        h_src = self.cross_attn1(h_src, h_tgt)
        h_src = self.cross_attn2(h_src, h_tgt)
        h_src = self.cross_attn3(h_src, h_tgt) # [B*K, H//16*W//16, 128]
        h3 = h_src.permute(0, 2, 1).reshape(b*k, self.channels, h//4, w//4) # [B*K, 128, H//16, W//16]

        # decoder
        h4 = self.up_conv1(h3) # [B*K, 128, H//8, W//8]
        h4 = torch.cat([h2, h4], dim=1) # [B*K, 256, H//8, W//8]
        h4 = self.res_dec1_1(h4) # [B*K, 128, H//8, W//8]
        h4 = self.res_dec1_2(h4) # [B*K, 128, H//8, W//8]
        h5 = self.up_conv2(h4) # [B*K, 128, H//8, W//8]
        h5 = torch.cat([h1, h5], dim=1) # [B*K, 256, H//4, W//4]
        out = self.res_dec2_1(h5) # [B*K, 128, H//4, W//4]
        out = self.res_dec2_2(out) # [B*K, 128, H//4, W//4]
        out = out.permute(0, 2, 3, 1).reshape(b, k, h, w, self.channels) # [B, K, H//4, W//4, 128]
        return out
