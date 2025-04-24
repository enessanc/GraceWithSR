"""
SR4 Eğitim Fonksiyonları
"""

import os
import gc
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import shutil

from ..loss import calculate_ssim_loss

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

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dir, test_func=None, scheduler=None):
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
        if test_func and ((epoch + 1) % 5 == 0 or epoch == 0):
            test_output_dir = os.path.join(save_dir, f'test_results_epoch_{epoch+1}')
            avg_psnr, avg_ssim = test_func(model, val_loader, device, test_output_dir)
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

    return best_val_loss, best_epoch 