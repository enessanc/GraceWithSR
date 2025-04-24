"""
FastSR Model Test Scripti - SR4
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from network import FastSR
from processing import VideoDataset
from utils import setup_cuda, test_model

if __name__ == "__main__":
    # CUDA ayarları
    device = setup_cuda()

    # Dizinler
    checkpoint_dir = os.path.join('checkpoints', 'sr4')
    processed_dir = "datasets/processed_videos"
    output_dir = "results/sr4/test_results"

    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Model parametreleri
    model_params = {
        'scale_factor': 1.875,
        'num_channels': 48,
        'num_blocks': 14,
        'dropout_rate': 0.2
    }

    # Test parametreleri
    test_params = {
        'batch_size': 4,
    }

    # Veri setini yükle
    print("Test veri seti yükleniyor...")
    test_paths = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir)
                 if f.endswith(('.mp4', '.avi', '.mov'))][-50:]  # Son 50 video

    print(f"Test set size: {len(test_paths)} videos")

    # Veri setini oluştur
    test_dataset = VideoDataset(test_paths)

    # DataLoader oluştur
    test_loader = DataLoader(test_dataset,
                            batch_size=test_params['batch_size'],
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # Modeli oluştur ve yükle
    model = FastSR(**model_params).to(device)
    
    # En iyi checkpoint'i yükle
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path} (Epoch {checkpoint['epoch']+1})")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Using untrained model.")

    # Test et
    print("Model test ediliyor...")
    avg_psnr, avg_ssim = test_model(model, test_loader, device, output_dir)
    
    print(f"Test tamamlandı! Ortalama PSNR: {avg_psnr:.4f}, Ortalama SSIM: {avg_ssim:.4f}")
