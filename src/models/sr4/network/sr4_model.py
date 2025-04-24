"""
FastSR Model - SR4 implementasyonu
SELayer, ChannelAttention, SpatialAttention, TextAttention, FastResidualBlock ve FastSR sınıflarını içerir.
"""

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

# Yazı tanıma için özel attention sınıfı
class TextAttention(nn.Module):
    def __init__(self, num_channels):
        super(TextAttention, self).__init__()

        # Daha verimli kanal yapısı (parametre sayısını azaltmak için)
        reduced_channels = max(16, num_channels // 4)

        # İlk katman: Girişi azaltıp metin özelliklerini çıkar
        self.feature_reduce = nn.Sequential(
            nn.Conv2d(num_channels, reduced_channels, kernel_size=1, padding=0), # 1x1 konv ile kanal sayısını azalt
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(True)
        )

        # Metin algılama - daha küçük ve hafif
        self.text_detect = nn.Conv2d(reduced_channels, 1, kernel_size=3, padding=1)

        # Aktivasyon fonksiyonu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Önce boyutu küçült
        features = self.feature_reduce(x)

        # Metin haritasını oluştur
        text_map = self.text_detect(features)

        # Dikkat ağırlıklarını hesapla
        attention = self.sigmoid(text_map)

        # Girişi ağırlıklandır - basit attention mekanizması
        enhanced = x * attention

        # Residual bağlantı ekle - orijinal değerler korunsun
        return x + enhanced

class FastResidualBlock(nn.Module):
    def __init__(self, num_channels, dropout_rate=0.2):
        super(FastResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SELayer(num_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Spatial attention ekle
        self.spatial_attention = SpatialAttention()

        # Text attention ekle - yazı tanıma için
        self.text_attention = TextAttention(num_channels)

    def forward(self, x):
        # Memory optimizasyonu için intermediate tensors'ları temizle
        with torch.amp.autocast(device_type='cuda'):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out = self.dropout(out)

            # Spatial attention uygula
            spatial_out = self.spatial_attention(out)
            # SpatialAttention çıktısının doğru kanal sayısını kontrol et
            if spatial_out.size(1) != out.size(1):
                # SpatialAttention sonrası kanal sayıları uyuşmuyorsa
                # Orijinal out tensörünü SpatialAttention ağırlıklarıyla çarp
                out = out * spatial_out
            else:
                out = spatial_out

            # Text attention uygula
            out = self.text_attention(out)

            # Residual bağlantı
            out = out + residual
            return F.relu(out)

class FastSR(nn.Module):
    def __init__(self, scale_factor=1.875, num_channels=48, num_blocks=10, dropout_rate=0.2):
        super(FastSR, self).__init__()

        # Memory optimizasyonu için autocast kullan
        self.use_autocast = True

        # Initial feature extraction - Daha derin başlangıç katmanı
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Residual blocks - Daha karmaşık bloklar
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(FastResidualBlock(num_channels, dropout_rate))
            # Her iki bloktan sonra channel attention ekle
            if _ % 2 == 1:
                res_blocks.append(ChannelAttention(num_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Global residual learning - Daha derin ara katman
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(True),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True)
        )

        # Progressive upsampling - Daha hassas upsampling
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
            nn.ReLU(True),
            nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((240, 240))
        )

    def forward(self, x):
        # Güncellenmiş autocast syntax'ı
        with torch.amp.autocast(device_type='cuda'):
            # Initial feature extraction
            out = self.conv_input(x)
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