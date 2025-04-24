"""
Loss functions for SR2 model
Includes:
- Perceptual Loss (VGG-based)
- Combined Loss (MSE + SSIM + Perceptual)
"""

import torch
import torch.nn as nn
import torchvision
from pytorch_msssim import ssim

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Sadece ilk birkaç katmanı kullan
        vgg = torchvision.models.vgg19(weights=None).features[:5].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        with torch.amp.autocast(device_type='cuda'):
            # Görüntüleri normalize et
            sr = (sr - 0.5) / 0.5
            hr = (hr - 0.5) / 0.5

            sr_features = self.vgg(sr)
            hr_features = self.vgg(hr)
            loss = self.criterion(sr_features, hr_features)

            # Loss'u normalize et
            loss = loss * 10.0  # Daha anlamlı değerler için ölçeklendir

            return loss

def calculate_ssim_loss(sr, hr):
    """Calculate SSIM loss"""
    # SSIM için Float32 kullan
    sr = sr.to(torch.float32)
    hr = hr.to(torch.float32)
    ssim_loss = 1 - ssim(sr, hr, data_range=1.0, size_average=True)
    return ssim_loss

class CombinedLoss(nn.Module):
    """Combined loss function (MSE + SSIM + Perceptual)"""
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # SSIM weight
        self.gamma = gamma  # Perceptual weight
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, sr, hr):
        # Loss hesaplamaları için Float32 kullan
        sr = sr.to(torch.float32)
        hr = hr.to(torch.float32)

        # MSE Loss
        mse = self.mse_loss(sr, hr)

        # SSIM Loss
        ssim = calculate_ssim_loss(sr, hr)

        # Perceptual Loss
        perceptual = self.perceptual_loss(sr, hr)

        # Combined loss
        total_loss = (self.alpha * mse + 
                     self.beta * ssim + 
                     self.gamma * perceptual)

        return total_loss 