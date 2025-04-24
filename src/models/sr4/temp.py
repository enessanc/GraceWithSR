"""
UYARI: Bu dosya artık kullanılmamaktadır.

Bu dosyadaki tüm kod, SR4 modeli için aşağıdaki modüler yapıya refaktör edilmiştir:

- network/sr4_model.py: Model sınıfları (FastSR, FastResidualBlock, vb.)
- processing/dataset.py: VideoDataset sınıfı
- loss/sr4_loss.py: Loss fonksiyonları
- utils/trainer.py: Eğitim fonksiyonları
- utils/tester.py: Test fonksiyonları
- config/model_config.py: Model konfigürasyonları
- train_model.py: Eğitim scripti
- test_model.py: Test scripti

Lütfen artık bu dosyaları kullanınız.
"""



# GPU Memory optimizasyonu
import torch
import gc
import time

# Tüm CUDA cihazlarını temizle
torch.cuda.empty_cache()
gc.collect()

# CUDA memory ayarları
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Google Drive'ı bağla
from google.colab import drive
drive.mount('/content/drive')

# Çalışma dizinini değiştir
os.chdir('/content/drive/MyDrive/Colab Notebooks')

# Memory durumunu kontrol et
print("GPU Memory Durumu:")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Cached Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from pytorch_msssim import ssim
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import shutil
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# SECTION 2: VERİ SETİ SINIFI
# =============================================================================
class VideoDataset(Dataset):
    """Video veri seti"""
    def __init__(self, video_paths, hr_size=(240, 240), lr_size=(128, 128), num_frames=4):
        self.video_paths = video_paths
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.num_frames = num_frames

        # Memory optimizasyonu için frame'leri önceden yükle
        self.frames_cache = {}
        self.current_video = None
        self.current_frames = None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Eğer aynı video ise cache'den al
        if video_path == self.current_video and self.current_frames is not None:
            frames_hr, frames_lr = self.current_frames
        else:
            # Yeni video için cache'i temizle
            self.current_video = video_path
            self.current_frames = None

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)

            frames_hr = []
            frames_lr = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Yüksek çözünürlüklü frame
                    frame_hr = cv2.resize(frame, self.hr_size)
                    frame_hr = frame_hr.astype(np.float32) / 255.0
                    frames_hr.append(frame_hr)

                    # Düşük çözünürlüklü frame
                    frame_lr = cv2.resize(frame, self.lr_size)
                    frame_lr = frame_lr.astype(np.float32) / 255.0
                    frames_lr.append(frame_lr)

            cap.release()

            if len(frames_hr) == 0:
                frames_hr = [np.zeros((self.hr_size[1], self.hr_size[0], 3), dtype=np.float32)
                            for _ in range(self.num_frames)]
                frames_lr = [np.zeros((self.lr_size[1], self.lr_size[0], 3), dtype=np.float32)
                            for _ in range(self.num_frames)]

            # Cache'e kaydet
            self.current_frames = (frames_hr, frames_lr)

        # Tensor'a çevir ve boyutları düzenle
        frames_hr = torch.from_numpy(np.array(frames_hr, dtype=np.float32))  # (num_frames, H, W, C)
        frames_lr = torch.from_numpy(np.array(frames_lr, dtype=np.float32))

        # Permute işlemi
        frames_hr = frames_hr.permute(0, 3, 1, 2)  # (num_frames, C, H, W)
        frames_lr = frames_lr.permute(0, 3, 1, 2)

        return frames_lr, frames_hr

# =============================================================================
# SECTION 3: MODEL SINIFI VE BİLEŞENLERİ
# =============================================================================
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
        # Input boyut kontrolü (debug için)
        # print(f"TextAttention input shape: {x.shape}")

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
            # SpatialAttention çıktısında kanal sayısı azalıyorsa, orijinal çıktıyı kullan
            # print(f"Spatial attention output shape: {spatial_out.shape}")
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

        # Memory optimizasyonu için autocast kullan (güncellenmiş syntax)
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

# =============================================================================
# SECTION 4: LOSS FONKSİYONLARI
# =============================================================================
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

# =============================================================================
# SECTION 5: EĞİTİM FONKSİYONLARI
# =============================================================================
def setup_cuda():
    """CUDA ayarlarını yapılandır"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def print_memory_status(prefix=""):
    """Memory durumunu detaylı olarak yazdır"""
    print(f"\n{prefix} Memory Durumu:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Cache Allocated: {torch.cuda.memory_cached(0) / 1024**3:.2f} GB")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """
    Tek bir eğitim epoch'u gerçekleştirir
    """
    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    running_perceptual = 0.0
    running_ssim = 0.0

    # Epoch başlangıcında GPU belleğini temizle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

    for batch_idx, (lr_videos, hr_videos) in enumerate(progress_bar):
        # Video batch işlemleri
        batch_size, seq_len, c, h, w = lr_videos.shape

        total_loss = 0
        total_l1 = 0
        total_perceptual = 0
        total_ssim = 0

        optimizer.zero_grad()

        # Videoyu kare kare işle
        for frame_idx in range(seq_len):
            # Kareleri al
            lr_frame = lr_videos[:, frame_idx].to(device)
            hr_frame = hr_videos[:, frame_idx].to(device)

            # Model ile tahmin
            with torch.amp.autocast(device_type='cuda'):
                sr_frame = model(lr_frame)
                loss = criterion(sr_frame, hr_frame)

            # Gradient hesapla
            loss.backward()

            # Kayıpları topla
            total_loss += loss.item()
            total_l1 += criterion.last_l1
            total_perceptual += criterion.last_perceptual
            total_ssim += criterion.last_ssim

            # Arada bir GPU belleğini temizle
            if frame_idx % 16 == 0 and frame_idx > 0:
                torch.cuda.empty_cache()

        # Karelerin ortalama kaybını al
        avg_loss = total_loss / seq_len
        avg_l1 = total_l1 / seq_len
        avg_perceptual = total_perceptual / seq_len
        avg_ssim = total_ssim / seq_len

        # Gradient'i uygula
        optimizer.step()

        # Loss değerini topla
        running_loss += avg_loss
        running_l1 += avg_l1
        running_perceptual += avg_perceptual
        running_ssim += avg_ssim

        # Progress bar'ı güncelle
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'l1': f'{avg_l1:.4f}',
            'perc': f'{avg_perceptual:.4f}',
            'ssim': f'{avg_ssim:.4f}'
        })

    # Epoch sonunda ortalama kayıpları hesapla
    train_loss = running_loss / len(train_loader)
    train_l1 = running_l1 / len(train_loader)
    train_perceptual = running_perceptual / len(train_loader)
    train_ssim = running_ssim / len(train_loader)

    return train_loss, train_l1, train_perceptual, train_ssim

def validate(model, val_loader, criterion, device):
    """
    Model validasyonu gerçekleştirir
    """
    model.eval()
    val_loss = 0.0
    val_l1 = 0.0
    val_perceptual = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for lr_videos, hr_videos in tqdm(val_loader, desc="Validating"):
            # Video boyutları
            batch_size, seq_len, c, h, w = lr_videos.shape

            # Her video için toplam kayıp
            batch_loss = 0
            batch_l1 = 0
            batch_perceptual = 0
            batch_ssim = 0

            # Her kareyi ayrı ayrı işle
            for frame_idx in range(seq_len):
                # Kareleri al
                lr_frame = lr_videos[:, frame_idx].to(device)
                hr_frame = hr_videos[:, frame_idx].to(device)

                # Modeli çalıştır
                with torch.amp.autocast(device_type='cuda'):
                    sr_frame = model(lr_frame)
                    loss = criterion(sr_frame, hr_frame)

                # Kayıpları topla
                batch_loss += loss.item()
                batch_l1 += criterion.last_l1
                batch_perceptual += criterion.last_perceptual
                batch_ssim += criterion.last_ssim

            # Ortalamaları hesapla ve topla
            val_loss += batch_loss / seq_len
            val_l1 += batch_l1 / seq_len
            val_perceptual += batch_perceptual / seq_len
            val_ssim += batch_ssim / seq_len

    # Tüm batches için ortalama değerleri hesapla
    val_loss /= len(val_loader)
    val_l1 /= len(val_loader)
    val_perceptual /= len(val_loader)
    val_ssim /= len(val_loader)

    return val_loss, val_l1, val_perceptual, val_ssim

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dir, scheduler=None):
    """
    Model eğitimi ve validasyon işlemi
    """
    best_val_loss = float('inf')
    best_epoch = 0

    # TensorBoard için log dizini
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        # Eğitim
        train_loss, train_l1, train_perceptual, train_ssim = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

        # Validasyon
        val_loss, val_l1, val_perceptual, val_ssim = validate(model, val_loader, criterion, device)

        # Test sonuçları (her 5 epoch'ta bir)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_output_dir = os.path.join(save_dir, f'test_results_epoch_{epoch+1}')
            avg_psnr, avg_ssim = test_model(model, val_loader, device, test_output_dir)
            writer.add_scalar('Test/PSNR', avg_psnr, epoch)
            writer.add_scalar('Test/SSIM', avg_ssim, epoch)

        # Learning rate düzenleme
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # TensorBoard günlükleri
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('L1/train', train_l1, epoch)
        writer.add_scalar('L1/val', val_l1, epoch)
        writer.add_scalar('Perceptual/train', train_perceptual, epoch)
        writer.add_scalar('Perceptual/val', val_perceptual, epoch)
        writer.add_scalar('SSIM/train', train_ssim, epoch)
        writer.add_scalar('SSIM/val', val_ssim, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f} (L1: {train_l1:.4f}, Perc: {train_perceptual:.4f}, SSIM: {train_ssim:.4f}) | "
              f"Val Loss: {val_loss:.6f} (L1: {val_l1:.4f}, Perc: {val_perceptual:.4f}, SSIM: {val_ssim:.4f})")

        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # En iyi model kaydı
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_epoch': best_epoch
            }

            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"Best model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}")

        # Son model kaydı
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, os.path.join(save_dir, 'last_model.pth'))

    writer.close()
    print(f"Training completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")

# =============================================================================
# SECTION 6: TEST FONKSİYONLARI
# =============================================================================
def save_comparison(lr_img, sr_img, hr_img, output_dir, index, epoch=None):
    """
    Compare and save the LR, SR, and HR images side by side
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Clip images to valid range for visualization
    lr_img = torch.clamp(lr_img, 0, 1)
    sr_img = torch.clamp(sr_img, 0, 1)
    hr_img = torch.clamp(hr_img, 0, 1)

    # Convert tensors to numpy arrays and scale to [0, 255]
    lr_np = (lr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sr_np = (sr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    hr_np = (hr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Get target size from HR image
    target_height, target_width = hr_np.shape[:2]

    # Resize LR image to match HR size
    lr_np = cv2.resize(lr_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Convert from RGB to BGR for OpenCV operations
    lr_bgr = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
    sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
    hr_bgr = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)

    # Apply color correction to SR image
    sr_hsv = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2HSV)

    # Increase saturation
    sr_hsv[:,:,1] = cv2.add(sr_hsv[:,:,1], 30)  # Increase saturation by 30

    # Apply contrast stretching to V channel
    v_channel = sr_hsv[:,:,2].astype(np.float32)  # Convert to float32 for calculations
    min_val = np.percentile(v_channel, 5)
    max_val = np.percentile(v_channel, 95)

    # Güvenli kontrast germe
    if max_val > min_val:
        v_channel = ((v_channel - min_val) * 255.0 / (max_val - min_val))
    else:
        # Eğer max_val = min_val ise, görüntü tamamen düz demektir
        # Bu durumda orijinal değerleri koru
        v_channel = sr_hsv[:,:,2].astype(np.float32)

    # Değerleri [0, 255] aralığına kırp ve uint8'e dönüştür
    v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)
    sr_hsv[:,:,2] = v_channel

    # Convert back to BGR
    sr_bgr = cv2.cvtColor(sr_hsv, cv2.COLOR_HSV2BGR)

    # Create a combined image
    h, w = target_height, target_width
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, :w] = lr_bgr
    combined[:, w:2*w] = sr_bgr
    combined[:, 2*w:] = hr_bgr

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Low Res', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Super Res', (w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'High Res', (2*w+10, 30), font, 1, (255, 255, 255), 2)

    # Save the combined image directly using OpenCV
    filename = f"comparison_{index}.png" if epoch is None else f"epoch_{epoch}_comparison_{index}.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, combined)

def test_model(model, test_loader, device, output_dir):
    """
    Test the model and save comparison images
    """
    model.eval()
    test_psnr = []
    test_ssim = []

    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'test_results')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Clear previous results
    os.makedirs(output_dir, exist_ok=True)

    video_output_dir = os.path.join(output_dir, 'videos')
    os.makedirs(video_output_dir, exist_ok=True)

    # Select a few random videos to create full output videos
    videos_to_save = random.sample(range(len(test_loader)), min(5, len(test_loader)))

    with torch.no_grad():
        for i, (lr_videos, hr_videos) in enumerate(test_loader):
            # Process each frame in the video batch
            batch_size, seq_len, c, h, w = lr_videos.shape

            # Initialize lists to store frames for video creation
            lr_frames = []
            sr_frames = []
            hr_frames = []

            for frame_idx in range(seq_len):
                # Process individual frames
                lr_frame = lr_videos[:, frame_idx].to(device)
                hr_frame = hr_videos[:, frame_idx].to(device)

                # Super-resolve the frame
                sr_frame = model(lr_frame)

                # Calculate metrics
                curr_psnr = calculate_psnr(sr_frame, hr_frame)
                curr_ssim = calculate_ssim(sr_frame, hr_frame)
                test_psnr.append(curr_psnr.item())
                test_ssim.append(curr_ssim.item())

                # Save comparison for the first frame
                if frame_idx == 0:
                    save_comparison(
                        lr_frame[0],
                        sr_frame[0],
                        hr_frame[0],
                        output_dir,
                        i
                    )

                # Store frames for video creation if this video is selected for saving
                if i in videos_to_save:
                    # Convert tensors to numpy for video saving
                    lr_np = (torch.clamp(lr_frame[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    sr_np = (torch.clamp(sr_frame[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    hr_np = (torch.clamp(hr_frame[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    # Get target size from HR image
                    target_height, target_width = hr_np.shape[:2]

                    # Resize LR image to match HR size
                    lr_np = cv2.resize(lr_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

                    # Convert from RGB to BGR for OpenCV
                    lr_np = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
                    sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
                    hr_np = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)

                    lr_frames.append(lr_np)
                    sr_frames.append(sr_np)
                    hr_frames.append(hr_np)

            # Create and save videos for selected test samples
            if i in videos_to_save and len(lr_frames) > 0:
                # Get dimensions from the HR frame
                height, width, _ = hr_frames[0].shape

                # Create side-by-side comparison video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(video_output_dir, f'video_comparison_{i}.mp4')

                # Create a writer for the combined video (3x width for LR, SR, HR side by side)
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width*3, height))

                for j in range(len(lr_frames)):
                    # Create a combined frame
                    combined_frame = np.zeros((height, width*3, 3), dtype=np.uint8)
                    combined_frame[:, :width] = lr_frames[j]
                    combined_frame[:, width:2*width] = sr_frames[j]
                    combined_frame[:, 2*width:] = hr_frames[j]

                    # Add frame labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined_frame, 'Low Res', (10, 30), font, 1, (255, 255, 255), 2)
                    cv2.putText(combined_frame, 'Super Res', (width+10, 30), font, 1, (255, 255, 255), 2)
                    cv2.putText(combined_frame, 'High Res', (2*width+10, 30), font, 1, (255, 255, 255), 2)

                    # Write the frame
                    video_writer.write(combined_frame)

                video_writer.release()
                print(f"Saved video comparison to {video_path}")

    # Calculate average metrics
    avg_psnr = sum(test_psnr) / len(test_psnr)
    avg_ssim = sum(test_ssim) / len(test_ssim)

    # Save metrics to a file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Test Results: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: PyTorch tensors with shape (B, C, H, W)
    Returns:
        PSNR value
    """
    # Ensure data range between 0 and 1
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # MSE for each image in batch
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    # PSNR calculation
    psnr = -10 * torch.log10(mse)

    # Average over batch
    return torch.mean(psnr)

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images
    Args:
        img1, img2: PyTorch tensors with shape (B, C, H, W)
    Returns:
        SSIM value
    """
    # Ensure data range between 0 and 1
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Calculate mean
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

    # Calculate variance and covariance
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Average over spatial dimensions and channels, then over batch
    return torch.mean(torch.mean(ssim_map, dim=[1, 2, 3]))

# =============================================================================
# SECTION 7: ÖRNEK KULLANIM
# =============================================================================
if __name__ == "__main__":
    # CUDA ayarları
    device = setup_cuda()

    # Checkpoint dizini
    save_dir = os.path.join('/content/drive/MyDrive/Colab Notebooks', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Model parametreleri - metin için derinliği artırdık
    model_params = {
        'scale_factor': 1.875,
        'num_channels': 48,
        'num_blocks': 14,
        'dropout_rate': 0.2
    }

    # Eğitim parametreleri
    train_params = {
        'batch_size': 8,  # Batch size'ı küçülttük
        'gradient_accumulation_steps': 4,  # Gradient accumulation'ı artırdık
        'learning_rate': 0.0001,  # Learning rate'i düşürdük
        'num_epochs': 100,
        'patience': 30,
        'max_grad_norm': 1.0  # Gradient clipping için
    }

    # Dizinler
    processed_dir = "/content/drive/MyDrive/Colab Notebooks/processed_videos"
    output_dir = "/content/drive/MyDrive/Colab Notebooks/results"

    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Veri setlerini yükle
    print("Veri setleri yükleniyor...")
    video_paths = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir)
                   if f.endswith(('.mp4', '.avi', '.mov'))]

    # Videoları rastgele karıştır
    np.random.seed(42)
    np.random.shuffle(video_paths)

    # Videoları setlere böl
    train_size = int(0.7 * len(video_paths))
    val_size = int(0.15 * len(video_paths))

    train_paths = video_paths[:train_size]
    val_paths = video_paths[train_size:train_size + val_size]
    test_paths = video_paths[train_size + val_size:]

    print(f"Train set size: {len(train_paths)} videos")
    print(f"Validation set size: {len(val_paths)} videos")
    print(f"Test set size: {len(test_paths)} videos")

    # Veri setlerini oluştur
    train_dataset = VideoDataset(train_paths)
    val_dataset = VideoDataset(val_paths)
    test_dataset = VideoDataset(test_paths)

    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset,
                             batch_size=train_params['batch_size'],
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True)  # Son eksik batch'i atla

    val_loader = DataLoader(val_dataset,
                           batch_size=train_params['batch_size'],
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True,
                           drop_last=True)  # Son eksik batch'i atla

    test_loader = DataLoader(test_dataset,
                            batch_size=train_params['batch_size'],
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)  # Son eksik batch'i atla

    # Model, loss ve optimizer
    model = FastSR(**model_params).to(device)

    # Loss fonksiyonu
    criterion = CombinedLoss().to(device)

    # Lion optimizer yerine Adam optimizer kullan
    optimizer = torch.optim.AdamW(model.parameters(),
                   lr=train_params['learning_rate'],
                   betas=(0.9, 0.999),
                   weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Eğitim

    print("Model eğitiliyor...")
    train_with_validation(
        model, train_loader, val_loader, criterion, optimizer,
        device, train_params['num_epochs'], save_dir, scheduler
    )

    # Eğitim sonuçlarını görselleştir
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()
 

    # Test
    print("Model test ediliyor...")
    checkpoint = torch.load('/content/drive/MyDrive/Colab Notebooks/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = test_model(model, test_loader, device, output_dir)