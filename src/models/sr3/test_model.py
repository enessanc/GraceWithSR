import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from network.model import FastSR
from processing.dataset import VideoDataset
from loss.losses import CombinedLoss
from utils.memory import setup_cuda, print_memory_status
from utils.visualization import save_comparison

def test_model(model, test_loader, criterion, device, output_dir):
    """
    Modeli test eder ve sonuçları kaydeder.
    
    Args:
        model: Eğitilmiş model
        test_loader: Test veri yükleyici
        criterion: Loss fonksiyonu
        device: Kullanılacak cihaz (cuda/cpu)
        output_dir: Çıktıların kaydedileceği dizin
    """
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0

    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)

    with torch.no_grad():
        for batch_idx, (lr_frames, hr_frames) in enumerate(tqdm(test_loader, desc='Testing')):
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)

            # Forward pass
            sr_frames = model(lr_frames)

            # Loss hesapla
            loss = criterion(sr_frames, hr_frames)
            total_loss += loss.item()

            # Her batch'ten bir örnek kaydet
            if batch_idx % 10 == 0:
                save_path = os.path.join(output_dir, 'comparisons', f'comparison_{batch_idx}.png')
                save_comparison(
                    lr_frames[0],  # İlk frame'i al
                    sr_frames[0],
                    hr_frames[0],
                    save_path
                )

    # Ortalama metrikleri hesapla
    avg_loss = total_loss / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)

    # Sonuçları kaydet
    results = {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }

    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        for metric, value in results.items():
            f.write(f'{metric}: {value:.6f}\n')

    return results

if __name__ == "__main__":
    # CUDA ayarları
    setup_cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_memory_status("Initial ")

    # Model ve veri seti parametreleri
    hr_size = (240, 240)
    lr_size = (128, 128)
    num_frames = 4
    batch_size = 1  # Test için batch size 1
    num_workers = 4

    # Modeli yükle
    model = FastSR(scale_factor=1.875, num_channels=48, num_blocks=10, dropout_rate=0.2)
    model = model.to(device)

    # Checkpoint'ten modeli yükle
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Loss fonksiyonu
    criterion = CombinedLoss(alpha=0.7, beta=0.2, gamma=0.1)

    # Test veri setini yükle
    test_video_paths = [...]  # Test video yolları
    test_dataset = VideoDataset(test_video_paths, hr_size, lr_size, num_frames)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Test et
    results = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        output_dir='test_results'
    )

    print("Test Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")
