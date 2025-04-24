"""
SR2 Model Training Script
Combines all components for training the improved SR model
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from network.sr2_model import FastSR
from loss.loss_functions import CombinedLoss
from processing.dataset import VideoDataset
from utils.trainer import (
    setup_cuda,
    print_memory_status,
    train_with_validation,
    test_model
)
from config.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    LOSS_CONFIG,
    DATASET_CONFIG,
    OPTIMIZER_CONFIG,
    SCHEDULER_CONFIG
)

def main():
    # CUDA ayarlarını yapılandır
    setup_cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_memory_status("Initial")

    # Dataset hazırlığı
    video_dir = "path/to/video/directory"  # Video dizinini belirtin
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    
    # Veri setini böl
    train_paths, temp_paths = train_test_split(video_paths, test_size=0.3, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

    # Dataset ve DataLoader oluştur
    train_dataset = VideoDataset(
        train_paths,
        hr_size=DATASET_CONFIG['hr_size'],
        lr_size=DATASET_CONFIG['lr_size'],
        num_frames=DATASET_CONFIG['num_frames']
    )
    val_dataset = VideoDataset(
        val_paths,
        hr_size=DATASET_CONFIG['hr_size'],
        lr_size=DATASET_CONFIG['lr_size'],
        num_frames=DATASET_CONFIG['num_frames']
    )
    test_dataset = VideoDataset(
        test_paths,
        hr_size=DATASET_CONFIG['hr_size'],
        lr_size=DATASET_CONFIG['lr_size'],
        num_frames=DATASET_CONFIG['num_frames']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model, loss ve optimizer oluştur
    model = FastSR(
        scale_factor=MODEL_CONFIG['scale_factor'],
        num_channels=MODEL_CONFIG['num_channels'],
        num_blocks=MODEL_CONFIG['num_blocks'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    ).to(device)

    criterion = CombinedLoss(
        alpha=LOSS_CONFIG['alpha'],
        beta=LOSS_CONFIG['beta'],
        gamma=LOSS_CONFIG['gamma']
    )

    optimizer = Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        betas=OPTIMIZER_CONFIG['betas'],
        eps=OPTIMIZER_CONFIG['eps'],
        weight_decay=OPTIMIZER_CONFIG['weight_decay']
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_CONFIG['mode'],
        factor=SCHEDULER_CONFIG['factor'],
        patience=SCHEDULER_CONFIG['patience'],
        verbose=SCHEDULER_CONFIG['verbose']
    )

    # Checkpoint dizini oluştur
    checkpoint_dir = "checkpoints/sr2"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Modeli eğit
    train_losses, val_losses = train_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        patience=TRAINING_CONFIG['patience'],
        save_dir=checkpoint_dir
    )

    # Test setinde değerlendir
    test_loss, test_psnr, test_ssim = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=os.path.join(checkpoint_dir, "test_results")
    )

    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test PSNR: {test_psnr:.2f} dB")
    print(f"Test SSIM: {test_ssim:.4f}")

if __name__ == "__main__":
    main()
