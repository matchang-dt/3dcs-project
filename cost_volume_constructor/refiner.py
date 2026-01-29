import lightning as L
import torch
from torch import nn

from extractor import WindowSelfAttention, WindowCrossAttention, patchify


class ResBlock4UNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        assert in_channels % 16 == 0
        assert out_channels % 16 == 0
        super().__init__()
        num_groups1 = in_channels // 16
        num_groups2 = out_channels // 16
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False, dtype=dtype)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, dtype=dtype)
        self.silu = nn.SiLU()
        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=in_channels, dtype=dtype)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=out_channels, dtype=dtype)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.gn1.weight, 1)
        nn.init.constant_(self.gn2.weight, 1)
        nn.init.constant_(self.gn1.bias, 0)
        nn.init.constant_(self.gn2.bias, 0)

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False, dtype=dtype)
            nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.gn1(x)
        out = self.silu(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = out + self.skip(x) # skip connection
        return out


class CostVolumeRefiner(L.LightningModule):
    def __init__(self, channels=128, feat_map_size=64, dtype=torch.float32):
        super().__init__()
        self.channels = channels
        self.dtype = dtype
        self.res_enc1_1 = ResBlock4UNet(channels * 2, channels, dtype)
        self.res_enc1_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_enc2_1 = ResBlock4UNet(channels, channels, dtype)
        self.res_enc2_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv2 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.up_conv1 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec1_1 = ResBlock4UNet(channels * 2, channels, dtype)
        self.res_dec1_2 = ResBlock4UNet(channels, channels, dtype)
        self.up_conv2 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec2_1 = ResBlock4UNet(channels * 2, channels, dtype)
        self.res_dec2_2 = ResBlock4UNet(channels, channels, dtype)
        self.self_attn = WindowSelfAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn1 = WindowCrossAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn2 = WindowCrossAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 windows: not swinT
        self.cross_attn3 = WindowCrossAttention(channels, window_size=feat_map_size//4, num_heads=1, dtype=dtype) # only 1 windows: not swinT

    def forward(self, x):
        # x: [B, K, H//4, W//4, d (128*2)]
        b, k, h, w, d = x.shape
        x.einops()
        x = x.permute(0, 1, 4, 2, 3).reshape(b * k, d, h, w) # [B, K, H//4, W//4, d] -> [B*K, d, H//4, W//4]
        # encoder
        h1 = self.res_enc1_1(x)
        h1 = self.res_enc1_2(h1)
        h2 = self.down_conv1(h1)
        h2 = self.res_enc2_1(h2)
        h2 = self.res_enc2_2(h2)
        h3 = self.down_conv2(h2) # [B*K, 128, H//16, W//16]
        # bottleneck
        h3 = patchify(h3.reshape(b, k, d // 2, h // 4, w // 4)) # [B*K, K=(src|tgt), H//16, W//16, 128]
        h_src = h3[:, 0, :, :, :].reshape(b * k, h // 4 * w // 4, d // 2) # [B*K, H//16, W//16, 128]
        h_tgt = h3[:, 1:, :, :, :].reshape(b * k, (k - 1) * h // 4 * w // 4, d // 2) # [B*K, K-1, H//16, W//16, 128]
        h_src = self.self_attn(h_src)
        h_src = self.cross_attn1(h_src, h_tgt)
        h_src = self.cross_attn2(h_src, h_tgt)
        h_src = self.cross_attn3(h_src, h_tgt) # [B*K, H//16*W//16, 128]
        h3 = h_src.permute(0, 2, 1).reshape(b * k, d // 2, h // 4, w // 4) # [B*K, 128, H//16, W//16]

        # decoder
        h4 = self.up_conv1(h3) # [B*K, 128, H//8, W//8]
        h4 = torch.cat([h2, h4], dim=1) # [B*K, 256, H//8, W//8]
        h4 = self.res_dec1_1(h4) # [B*K, 128, H//8, W//8]
        h4 = self.res_dec1_2(h4) # [B*K, 128, H//8, W//8]
        h5 = self.up_conv2(h4) # [B*K, 128, H//8, W//8]
        h5 = torch.cat([h1, h5], dim=1) # [B*K, 256, H//4, W//4]
        out = self.res_dec2_1(h5) # [B*K, 128, H//4, W//4]
        out = self.res_dec2_2(out) # [B*K, 128, H//4, W//4]
        out = out.permute(0, 2, 3, 1).reshape(b, k, h, w, d // 2) # [B, K, H//4, W//4, 128]
        return out