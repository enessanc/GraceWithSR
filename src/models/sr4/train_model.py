"""
FastSR Model Eğitim Scripti - SR4
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from network import FastSR
from processing import VideoDataset
from loss import CombinedLoss
from utils import setup_cuda, train_with_validation, test_model

if __name__ == "__main__":
    # CUDA ayarları
    device = setup_cuda()

    # Checkpoint dizini
    save_dir = os.path.join('checkpoints', 'sr4')
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
    processed_dir = "datasets/processed_videos"
    output_dir = "results/sr4"

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

    # Optimizer
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
    best_val_loss, best_epoch = train_with_validation(
        model, train_loader, val_loader, criterion, optimizer,
        device, train_params['num_epochs'], save_dir, test_model, scheduler
    )

    print(f"Eğitim tamamlandı! En iyi validasyon loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")
