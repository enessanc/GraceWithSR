"""
SR2 Model Architecture
Improved version of SR1 with:
- SE (Squeeze-and-Excitation) layers
- Batch Normalization
- Dropout
- Progressive upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FastResidualBlock(nn.Module):
    """Improved residual block with SE layer and dropout"""
    def __init__(self, num_channels, dropout_rate=0.1):
        super(FastResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SELayer(num_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out = self.dropout(out)
            out = out + residual
            return F.relu(out)

class FastSR(nn.Module):
    """Improved SR model with progressive upsampling"""
    def __init__(self, scale_factor=1.875, num_channels=24, num_blocks=8, dropout_rate=0.1):
        super(FastSR, self).__init__()

        # Memory optimizasyonu için autocast kullan
        self.autocast = torch.amp.autocast('cuda')

        # Initial feature extraction
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(FastResidualBlock(num_channels, dropout_rate))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Global residual learning
        self.conv_mid = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # Progressive upsampling
        self.upscale = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Dropout2d(p=dropout_rate/2),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((240, 240))
        )

    def forward(self, x):
        with self.autocast:
            # Initial feature extraction
            out = F.relu(self.conv_input(x))
            out = self.dropout(out)

            # Residual blocks
            residual = out
            out = self.res_blocks(out)
            out = self.conv_mid(out)
            out = self.dropout(out)
            out = out + residual

            # Final upsampling
            out = self.upscale(out)

            return out 