import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
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

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x * self.sigmoid(x)

class FastResidualBlock(nn.Module):
    def __init__(self, num_channels, dropout_rate=0.2):
        super(FastResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SELayer(num_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        with torch.cuda.amp.autocast():
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out = self.dropout(out)
            out = self.spatial_attention(out)
            out = out + residual
            return F.relu(out)

class FastSR(nn.Module):
    def __init__(self, scale_factor=1.875, num_channels=48, num_blocks=10, dropout_rate=0.2):
        super(FastSR, self).__init__()
        self.autocast = torch.amp.autocast('cuda')

        # Initial feature extraction
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(FastResidualBlock(num_channels, dropout_rate))
            if _ % 2 == 1:
                res_blocks.append(ChannelAttention(num_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Global residual learning
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(True),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True)
        )

        # Progressive upsampling
        self.upscale = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.Dropout2d(p=dropout_rate/2),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True)
        )

        # Final reconstruction
        self.conv_output = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        with self.autocast:
            # Initial feature extraction
            x = self.conv_input(x)
            x = self.dropout(x)

            # Residual blocks
            residual = x
            x = self.res_blocks(x)
            x = x + residual

            # Global residual learning
            x = self.conv_mid(x)

            # Progressive upsampling
            x = self.upscale(x)

            # Final reconstruction
            x = self.conv_output(x)
            return x 