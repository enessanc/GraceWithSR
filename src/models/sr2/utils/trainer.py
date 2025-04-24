"""
Training utilities for SR2 model
Includes:
- Memory optimization
- Early stopping
- Training and validation loops
"""

import os
import time
from datetime import timedelta
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

def setup_cuda():
    """Setup CUDA environment"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def print_memory_status(prefix=""):
    """Print GPU memory status"""
    print(f"{prefix}GPU Memory Status:")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Cached Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}')

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Forward pass
        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Memory cleanup
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, scheduler,
                         device, num_epochs, patience, save_dir='checkpoints'):
    """Train model with validation and early stopping"""
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, num_epochs)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Print progress
        elapsed = time.time() - start_time
        eta = elapsed * (num_epochs - epoch) / epoch
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - "
              f"ETA: {timedelta(seconds=int(eta))}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

    return train_losses, val_losses

def test_model(model, test_loader, criterion, device, save_dir='test_results'):
    """Test the model"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            total_loss += loss.item()

            # Calculate PSNR and SSIM
            for i in range(sr_imgs.size(0)):
                sr_img = sr_imgs[i].cpu().numpy()
                hr_img = hr_imgs[i].cpu().numpy()
                
                # PSNR
                mse = np.mean((sr_img - hr_img) ** 2)
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                total_psnr += psnr

                # SSIM
                ssim_val = ssim(sr_img, hr_img, data_range=1.0)
                total_ssim += ssim_val

    avg_loss = total_loss / len(test_loader)
    avg_psnr = total_psnr / (len(test_loader) * test_loader.batch_size)
    avg_ssim = total_ssim / (len(test_loader) * test_loader.batch_size)

    print(f"Test Results:")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    return avg_loss, avg_psnr, avg_ssim 