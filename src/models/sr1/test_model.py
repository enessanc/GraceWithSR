import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from network.sr1_model import FastSR
from processing.dataset import ProcessedVideoDataset
from config.config import model_params, dataset_params

def test_model(model, test_loader, device, output_dir):
    """Modeli test seti üzerinde değerlendir"""
    model.eval()
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for lr_videos, hr_videos in tqdm(test_loader, desc='Testing'):
            lr_videos = lr_videos.to(device)
            hr_videos = hr_videos.to(device)

            outputs = model(lr_videos)

            # Her video için PSNR ve SSIM hesapla
            for i in range(outputs.size(0)):
                output = outputs[i].cpu().numpy()
                target = hr_videos[i].cpu().numpy()

                # Her frame için metrikleri hesapla
                for j in range(output.shape[0]):
                    # Channel last formatına çevir
                    output_frame = np.transpose(output[j], (1, 2, 0))
                    target_frame = np.transpose(target[j], (1, 2, 0))

                    # PSNR hesapla
                    psnr = peak_signal_noise_ratio(target_frame, output_frame)
                    psnr_values.append(psnr)

                    # SSIM hesapla
                    ssim = structural_similarity(target_frame, output_frame, multichannel=True)
                    ssim_values.append(ssim)

    # Ortalama metrikleri hesapla
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')

    return avg_psnr, avg_ssim

def main():
    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Modeli yükle
    model = FastSR(**model_params).to(device)
    checkpoint = torch.load(os.path.join('checkpoints', 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test veri setini yükle
    test_dir = "VideoDataSet"
    output_dir = "results"
    video_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                  if f.endswith(('.mp4', '.avi', '.mov'))]

    test_dataset = ProcessedVideoDataset(video_paths, **dataset_params)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Modeli test et
    avg_psnr, avg_ssim = test_model(model, test_loader, device, output_dir)

if __name__ == "__main__":
    main()
