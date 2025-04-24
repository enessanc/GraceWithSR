import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

def calculate_ssim_loss(sr, hr):
    # SSIM için Float32 kullan
    sr = sr.to(torch.float32)
    hr = hr.to(torch.float32)
    return 1 - ssim(sr, hr, data_range=1.0, size_average=True)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return self.criterion(sr_features, hr_features)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, sr, hr):
        # Loss hesaplamaları için Float32 kullan
        sr = sr.to(torch.float32)
        hr = hr.to(torch.float32)

        l1 = self.l1_loss(sr, hr)
        ssim = calculate_ssim_loss(sr, hr)
        perceptual = self.perceptual_loss(sr, hr)

        return (self.alpha * l1 + 
                self.beta * ssim + 
                self.gamma * perceptual) 