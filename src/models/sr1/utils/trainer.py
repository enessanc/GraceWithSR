import os
import time
from datetime import timedelta
import torch
from tqdm import tqdm

def train_with_validation(model, train_loader, val_loader, criterion, optimizer,
                         device, num_epochs, patience, save_dir='checkpoints'):
    """Early stopping ile eğitim"""
    # Dizinleri oluştur
    os.makedirs(save_dir, exist_ok=True)

    # Modeli GPU'ya taşı
    model = model.to(device)

    # Early stopping için değişkenler
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Eğitim başlangıç zamanı
    start_time = time.time()

    # Eğitim döngüsü
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        # Progress bar için
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for lr_videos, hr_videos in train_pbar:
            # Verileri GPU'ya taşı
            lr_videos = lr_videos.to(device)
            hr_videos = hr_videos.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(lr_videos)
            loss = criterion(outputs, hr_videos)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Loss hesaplama
            train_loss += loss.item()
            train_batches += 1

            # Progress bar'ı güncelle
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Ortalama train loss
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        # Validation için progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')

        with torch.no_grad():
            for lr_videos, hr_videos in val_pbar:
                # Verileri GPU'ya taşı
                lr_videos = lr_videos.to(device)
                hr_videos = hr_videos.to(device)

                # Forward pass
                outputs = model(lr_videos)
                loss = criterion(outputs, hr_videos)

                # Loss hesaplama
                val_loss += loss.item()
                val_batches += 1

                # Progress bar'ı güncelle
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Ortalama validation loss
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # Eğitim süresini hesapla
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = avg_time_per_epoch * remaining_epochs

        # Eğitim bilgilerini yazdır
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Geçen Süre: {timedelta(seconds=int(elapsed_time))}')
        print(f'Tahmini Kalan Süre: {timedelta(seconds=int(estimated_remaining_time))}')

        # GPU bellek kullanımını göster
        if torch.cuda.is_available():
            print(f'GPU Bellek Kullanımı: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')

        # Checkpoint kaydetme
        if (epoch + 1) % 5 == 0:  # Her 5 epoch'ta bir
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

        # Early stopping kontrolü
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # En iyi modeli kaydet
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping: {patience} epoch boyunca iyileşme olmadı')
                break

    return train_losses, val_losses 