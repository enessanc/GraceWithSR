import torch
import torch.nn as nn
import torch.nn.functional as F

class FastResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(FastResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out

class FastSR(nn.Module):
    def __init__(self, scale_factor=4, num_channels=32):
        super(FastSR, self).__init__()
        self.scale_factor = scale_factor

        # Feature extraction
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[FastResidualBlock(num_channels) for _ in range(4)]
        )

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

        # Final convolution
        self.conv2 = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [batch, frames, channels, height, width]
        batch_size, num_frames, channels, height, width = x.size()

        # Her frame'i ayrı ayrı işle
        outputs = []
        for i in range(num_frames):
            # Frame'i al
            frame = x[:, i]  # [batch, channels, height, width]

            # Feature extraction
            out = F.relu(self.conv1(frame))

            # Residual blocks
            out = self.res_blocks(out)

            # Upsampling
            out = self.upsample(out)

            # Final convolution
            out = self.conv2(out)

            outputs.append(out)

        # Output'ları stack et
        return torch.stack(outputs, dim=1)  # [batch, frames, channels, height, width] 