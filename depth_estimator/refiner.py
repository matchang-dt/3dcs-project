import lightning as L
import torch
from torch import nn

from utils import WindowSelfAttention, WindowCrossAttention, ResBlock4UNet, patchify


class DepthRefiner(L.LightningModule):
    def __init__(self, channels=128, feat_map_size=256, dtype=torch.float32):
        assert feat_map_size % 16 == 0
        super().__init__()
        self.to(dtype)
        self.channels = channels

        self.in_conv = nn.Conv2d(channels+4, channels, 3, stride=1, padding=1, bias=False, dtype=dtype)
        self.res_enc1_1 = ResBlock4UNet(channels, channels, dtype)
        self.res_enc1_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_enc2_1 = ResBlock4UNet(channels, channels, dtype)
        self.res_enc2_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv2 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_enc3_1 = ResBlock4UNet(channels, channels, dtype)
        self.res_enc3_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv3 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_enc4_1 = ResBlock4UNet(channels, channels, dtype)
        self.res_enc4_2 = ResBlock4UNet(channels, channels, dtype)
        self.down_conv4 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False, dtype=dtype)
        self.up_conv1 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec1_1 = ResBlock4UNet(channels*2, channels, dtype)
        self.res_dec1_2 = ResBlock4UNet(channels, channels, dtype)
        self.up_conv2 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec2_1 = ResBlock4UNet(channels*2, channels, dtype)
        self.res_dec2_2 = ResBlock4UNet(channels, channels, dtype)
        self.up_conv3 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec3_1 = ResBlock4UNet(channels*2, channels, dtype)
        self.res_dec3_2 = ResBlock4UNet(channels, channels, dtype)
        self.up_conv4 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False, dtype=dtype)
        self.res_dec4_1 = ResBlock4UNet(channels*2, channels, dtype)
        self.res_dec4_2 = ResBlock4UNet(channels, channels, dtype)
        self.self_attn = WindowSelfAttention(channels, window_size=feat_map_size//16, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn1 = WindowCrossAttention(channels, window_size=feat_map_size//16, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn2 = WindowCrossAttention(channels, window_size=feat_map_size//16, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.cross_attn3 = WindowCrossAttention(channels, window_size=feat_map_size//16, num_heads=1, dtype=dtype) # only 1 window: not swinT
        self.last_conv = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=True, dtype=dtype) # ch 128 -> 1

        nn.init.kaiming_normal_(self.in_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.down_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.down_conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.down_conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.down_conv4.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_conv4.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.last_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.last_conv.bias, 0)

    def forward(self, x):
        # x: [B, K, H, W, d (128 + 4)]
        b, k, H, W, d = x.shape
        assert d == self.channels + 4
        x = x.reshape(-1, H, W, d).permute(0, 3, 1, 2) # [B*K, d (128 + 4), H, W]
        x = self.in_conv(x) # [B*K, 128, H, W]
        # encoder
        h1 = self.res_enc1_1(x)
        h1 = self.res_enc1_2(h1)
        h2 = self.down_conv1(h1)
        h2 = self.res_enc2_1(h2)
        h2 = self.res_enc2_2(h2)
        h3 = self.down_conv2(h2)
        h3 = self.res_enc3_1(h3)
        h3 = self.res_enc3_2(h3)
        h4 = self.down_conv3(h3)
        h4 = self.res_enc4_1(h4)
        h4 = self.res_enc4_2(h4)
        h5 = self.down_conv4(h4) # [B*K, 128, H//16, W//16]

        # bottleneck
        h5 = patchify(h5.reshape(b, k, self.channels, H//16, W//16)) # [B*K, K=(src|tgt), H//16, W//16, 128]
        h_src = h5[:, 0, :, :, :].reshape(b*k, H//16 * W//16, self.channels) # [B*K, H//16, W//16, 128]
        h_tgt = h5[:, 1:, :, :, :].reshape(b*k, (k-1) * H//16 * W//16, self.channels) # [B*K, (K-1) * H//16 * W//16, 128]
        h_src = self.self_attn(h_src)
        h_src = self.cross_attn1(h_src, h_tgt)
        h_src = self.cross_attn2(h_src, h_tgt)
        h_src = self.cross_attn3(h_src, h_tgt) # [B*K, H//16*W//16, 128]
        h5 = h_src.permute(0, 2, 1).reshape(b * k, self.channels, H//16, W//16) # [B*K, 128, H//16, W//16]

        # decoder
        h6 = self.up_conv1(h5) # [B*K, 128, H//8, W//8]
        h6 = torch.cat([h4, h6], dim=1) # [B*K, 256, H//8, W//8]
        h6 = self.res_dec1_1(h6) # [B*K, 128, H//8, W//8]
        h6 = self.res_dec1_2(h6) # [B*K, 128, H//8, W//8]
        h7 = self.up_conv2(h6) # [B*K, 128, H//8, W//8]
        h7 = torch.cat([h3, h7], dim=1) # [B*K, 256, H//4, W//4]
        h7 = self.res_dec2_1(h7) # [B*K, 128, H//4, W//4]
        h7 = self.res_dec2_2(h7) # [B*K, 128, H//4, W//4]
        h8 = self.up_conv3(h7) # [B*K, 128, H//4, W//4]
        h8 = torch.cat([h2, h8], dim=1) # [B*K, 256, H//2, W//2]
        h8 = self.res_dec3_1(h8) # [B*K, 128, H//2, W//2]
        h8 = self.res_dec3_2(h8) # [B*K, 128, H//2, W//2]
        h9 = self.up_conv4(h8) # [B*K, 128, H//2, W//2]
        h9 = torch.cat([h1, h9], dim=1) # [B*K, 256, H, W]
        h9 = self.res_dec4_1(h9) # [B*K, 128, H, W]
        h9 = self.res_dec4_2(h9) # [B*K, 128, H, W]
        out = self.last_conv(h9) # [B*K, 1, H, W]
        out = out.permute(0, 2, 3, 1).reshape(b, k, H, W) # [B, K, H, W]
        return out # [B, K, H, W]
