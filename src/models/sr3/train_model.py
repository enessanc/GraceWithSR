import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from lion_pytorch import Lion

from network.model import FastSR
from processing.dataset import VideoDataset
from loss.losses import CombinedLoss
from utils.memory import setup_cuda, print_memory_status
from utils.visualization import save_comparison

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    with tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}') as pbar:
        for batch_idx, (lr_frames, hr_frames) in enumerate(pbar):
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)

            # Forward pass
            optimizer.zero_grad()
            sr_frames = model(lr_frames)

            # Loss hesapla
            loss = criterion(sr_frames, hr_frames)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Memory optimizasyonu
            torch.cuda.empty_cache()

            # Progress bar'ı güncelle
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

    # Epoch sonu işlemleri
    scheduler.step()
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} completed in {epoch_time:.2f}s - Average Loss: {avg_loss:.6f}')

    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for lr_frames, hr_frames in val_loader:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)

            # Forward pass
            sr_frames = model(lr_frames)

            # Loss hesapla
            loss = criterion(sr_frames, hr_frames)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_with_validation(model, train_loader, val_loader, criterion, optimizer,
                         device, num_epochs, patience, save_dir='checkpoints'):
    # Checkpoint klasörünü oluştur
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # Eğitim
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, num_epochs)

        # Validasyon
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.6f}')

        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        # Early stopping kontrolü
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch} epochs')
            break

    return best_val_loss

if __name__ == "__main__":
    # CUDA ayarları
    setup_cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_memory_status("Initial ")

    # Model ve veri seti parametreleri
    hr_size = (240, 240)
    lr_size = (128, 128)
    num_frames = 4
    batch_size = 4
    num_workers = 4

    # Model oluştur
    model = FastSR(scale_factor=1.875, num_channels=48, num_blocks=10, dropout_rate=0.2)
    model = model.to(device)

    # Loss ve optimizer
    criterion = CombinedLoss(alpha=0.7, beta=0.2, gamma=0.1)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Veri setlerini yükle
    train_video_paths = [...]  # Eğitim video yolları
    val_video_paths = [...]    # Validasyon video yolları

    train_dataset = VideoDataset(train_video_paths, hr_size, lr_size, num_frames)
    val_dataset = VideoDataset(val_video_paths, hr_size, lr_size, num_frames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Eğitim
    best_val_loss = train_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        patience=10,
        save_dir='checkpoints'
    )

    print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
