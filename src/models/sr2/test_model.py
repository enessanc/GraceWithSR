"""
SR2 Model Testing Script
Evaluates the trained model and visualizes results
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from network.sr2_model import FastSR
from processing.dataset import VideoDataset
from config.config import (
    MODEL_CONFIG,
    DATASET_CONFIG,
    TRAINING_CONFIG
)

def visualize_results(lr_img, sr_img, hr_img, save_path):
    """Görüntüleri karşılaştırmalı olarak görselleştir"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(lr_img)
    plt.title('Low Resolution')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sr_img)
    plt.title('Super Resolution')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(hr_img)
    plt.title('High Resolution')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader, device, save_dir):
    """Modeli test setinde değerlendir ve sonuçları görselleştir"""
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(test_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Model çıktısını al
            sr_imgs = model(lr_imgs)
            
            # Her bir görüntü için metrikleri hesapla
            for i in range(sr_imgs.size(0)):
                sr_img = sr_imgs[i].cpu().numpy().transpose(1, 2, 0)
                hr_img = hr_imgs[i].cpu().numpy().transpose(1, 2, 0)
                lr_img = lr_imgs[i].cpu().numpy().transpose(1, 2, 0)
                
                # PSNR ve SSIM hesapla
                psnr_val = psnr(hr_img, sr_img, data_range=1.0)
                ssim_val = ssim(hr_img, sr_img, data_range=1.0, channel_axis=2)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                num_samples += 1
                
                # Her 10 görüntüde bir sonuçları görselleştir
                if batch_idx % 10 == 0 and i == 0:
                    save_path = os.path.join(save_dir, f'result_{batch_idx}_{i}.png')
                    visualize_results(lr_img, sr_img, hr_img, save_path)
    
    # Ortalama metrikleri hesapla
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_psnr, avg_ssim

def main():
    # CUDA ayarlarını yapılandır
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test dataset ve loader oluştur
    test_dir = "path/to/test/videos"  # Test video dizinini belirtin
    test_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.mp4', '.avi'))]
    
    test_dataset = VideoDataset(
        test_paths,
        hr_size=DATASET_CONFIG['hr_size'],
        lr_size=DATASET_CONFIG['lr_size'],
        num_frames=DATASET_CONFIG['num_frames']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Modeli yükle
    model = FastSR(
        scale_factor=MODEL_CONFIG['scale_factor'],
        num_channels=MODEL_CONFIG['num_channels'],
        num_blocks=MODEL_CONFIG['num_blocks'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    ).to(device)
    
    # Checkpoint'ten modeli yükle
    checkpoint_path = "checkpoints/sr2/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Modeli değerlendir
    save_dir = "results/sr2"
    avg_psnr, avg_ssim = evaluate_model(model, test_loader, device, save_dir)
    
    print(f"\nTest Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
