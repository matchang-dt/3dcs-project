import lightning as L
import torch
import torch.nn as nn


class ResBlock4Extractor(L.LightningModule):
    def __init__(self, in_channels, out_channels, stride=1, dtype=torch.float32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, dtype=dtype)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, dtype=dtype)
        self.silu = nn.SiLU()
        self.bn1 = nn.BatchNorm2d(out_channels, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(out_channels, dtype=dtype)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
            nn.init.kaiming_normal_(self.skip.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.skip(x) # skip connection
        out = self.silu(out)
        return out


class ResBlock4UNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        assert in_channels % 16 == 0, "in_channels must be divisible by 16 for group normalization"
        assert out_channels % 16 == 0, "out_channels must be divisible by 16 for group normalization"
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
            nn.init.kaiming_normal_(self.skip.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.gn1(x)
        out = self.silu(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.silu(out)
        out = self.conv2(out)

        out = out + self.skip(x) # skip connection
        return out
