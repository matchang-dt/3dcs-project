from math import prod

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(L.LightningModule):
    def __init__(self, in_channels, out_channels, stride=1, dtype=torch.float32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, dtype=dtype)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, dtype=dtype)
        self.relu = nn.ReLU()
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
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.skip(x) # skip connection
        out = self.relu(out)
        
        return out

        
class CNNExtractor(L.LightningModule): # [B, K, 3, H, W] -> [B, K, 128, H//4, W//4]
    def __init__(self, out_channels=128, dtype=torch.float32):
        super().__init__()
        self.to(dtype)
        hidden_channels1 = out_channels // 4
        hidden_channels2 = out_channels // 2
        self.conv1 = nn.Conv2d(3, hidden_channels1, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(hidden_channels1, dtype=dtype)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.relu = nn.ReLU()
        self.res1 = ResBlock(hidden_channels1, hidden_channels1, dtype=dtype)
        self.res2 = ResBlock(hidden_channels1, hidden_channels1, dtype=dtype)
        self.res3 = ResBlock(hidden_channels1, hidden_channels1, dtype=dtype)
        self.res4 = ResBlock(hidden_channels1, hidden_channels2, stride=2, dtype=dtype)
        self.res5 = ResBlock(hidden_channels2, hidden_channels2, dtype=dtype)
        self.res6 = ResBlock(hidden_channels2, out_channels, stride=2, dtype=dtype)
        self.proj = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=1, stride=1, padding=0, bias=True, dtype=dtype
        )
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        # x: [B, K, 3, H, W] 
        if x.dim() == 4:
            b = 1
            k = x.shape[0]
        else:
            b, k = x.shape[:2]
            x = x.view(b * k, *x.shape[-3:]) # [B * K, 3, H, W]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.proj(out)
        return out.reshape(b, k, *out.shape[-3:]) # [B, K, 128, H//4, W//4]
