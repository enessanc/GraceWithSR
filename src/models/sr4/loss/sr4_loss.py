"""
SR4 Loss Fonksiyonları
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_msssim import ssim

def calculate_ssim_loss(sr, hr):
    # SSIM için Float32 kullan
    sr = sr.to(torch.float32)
    hr = hr.to(torch.float32)

    try:
        ssim_val = ssim(sr, hr, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val
        return ssim_loss
    except Exception as e:
        return torch.tensor(0.5, device=sr.device)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        try:
            # Sadece ilk birkaç katmanı kullan (derinlik belirleme)
            vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features[:10].eval()
        except:
            vgg = torchvision.models.vgg19(pretrained=True).features[:10].eval()

        # model parametrelerini freeze et
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        # NaN kontrolü
        if torch.isnan(sr).any() or torch.isnan(hr).any():
            sr = torch.nan_to_num(sr)
            hr = torch.nan_to_num(hr)

        # İlk olarak batch ve kanal boyutları kontrol edilmeli
        if sr.dim() == 5:  # 5D tensor: [B, T, C, H, W]
            b, t, c, h, w = sr.size()
            sr = sr.view(-1, c, h, w)  # 4D tensor: [B*T, C, H, W]
            hr = hr.view(-1, c, h, w)

        try:
            with torch.amp.autocast(device_type='cuda'):
                # Görüntüleri normalize et
                sr = (sr - 0.5) / 0.5  # -1 ile 1 arasına normalize et
                hr = (hr - 0.5) / 0.5

                # Feature extract
                sr_features = self.vgg(sr)
                hr_features = self.vgg(hr)

                # Loss hesapla
                loss = self.criterion(sr_features, hr_features)

                # Loss'u anlamlı değerlere ölçeklendir - çok daha büyük değil
                loss = loss * 0.01

                return loss

        except Exception as e:
            # Hata durumunda sabit değer döndür
            return torch.tensor(0.1, device=sr.device)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.15, gamma=0.05):
        super(CombinedLoss, self).__init__()
        # Metin algılama için perceptual loss ağırlığını arttırıyoruz
        self.alpha = alpha  # L1 loss ağırlığı
        self.beta = beta    # Perceptual loss ağırlığı - metin için daha yüksek
        self.gamma = gamma  # SSIM loss ağırlığı
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.last_l1 = 0.0
        self.last_perceptual = 0.0
        self.last_ssim = 0.0

    def forward(self, sr, hr):
        # Loss hesaplamaları için Float32 kullan
        sr = sr.to(torch.float32)
        hr = hr.to(torch.float32)

        # NaN kontrolü
        if torch.isnan(sr).any() or torch.isnan(hr).any():
            sr = torch.nan_to_num(sr)
            hr = torch.nan_to_num(hr)

        try:
            # L1 loss
            l1 = self.l1_loss(sr, hr)

            # Perceptual loss - 0-1 arasına normalize et
            perceptual = self.perceptual_loss(sr, hr)
            perceptual = torch.clamp(perceptual, 0, 1)

            # SSIM loss - yumuşak normalizasyon
            ssim_loss = calculate_ssim_loss(sr, hr)
            ssim_loss = torch.sigmoid(ssim_loss)

            # Toplam loss
            total_loss = self.alpha * l1 + self.beta * perceptual + self.gamma * ssim_loss

            # Loss değerlerini kaydet
            self.last_l1 = l1.item()
            self.last_perceptual = perceptual.item()
            self.last_ssim = ssim_loss.item()

            return total_loss

        except Exception as e:
            return torch.tensor(0.5, device=sr.device) 