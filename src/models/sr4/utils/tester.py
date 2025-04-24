"""
SR4 Test Fonksiyonları
"""

import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from tqdm import tqdm

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: PyTorch tensors with shape (B, C, H, W)
    Returns:
        PSNR value
    """
    # Ensure data range between 0 and 1
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # MSE for each image in batch
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    # PSNR calculation
    psnr = -10 * torch.log10(mse)

    # Average over batch
    return torch.mean(psnr)

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images
    Args:
        img1, img2: PyTorch tensors with shape (B, C, H, W)
    Returns:
        SSIM value
    """
    # Ensure data range between 0 and 1
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Calculate mean
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

    # Calculate variance and covariance
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Average over spatial dimensions and channels, then over batch
    return torch.mean(torch.mean(ssim_map, dim=[1, 2, 3]))

def save_comparison(lr_img, sr_img, hr_img, output_dir, index, epoch=None):
    """
    Compare and save the LR, SR, and HR images side by side
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Clip images to valid range for visualization
    lr_img = torch.clamp(lr_img, 0, 1)
    sr_img = torch.clamp(sr_img, 0, 1)
    hr_img = torch.clamp(hr_img, 0, 1)

    # Convert tensors to numpy arrays and scale to [0, 255]
    lr_np = (lr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sr_np = (sr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    hr_np = (hr_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Get target size from HR image
    target_height, target_width = hr_np.shape[:2]

    # Resize LR image to match HR size
    lr_np = cv2.resize(lr_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Convert from RGB to BGR for OpenCV operations
    lr_bgr = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
    sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
    hr_bgr = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)

    # Apply color correction to SR image
    sr_hsv = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2HSV)

    # Increase saturation
    sr_hsv[:,:,1] = cv2.add(sr_hsv[:,:,1], 30)  # Increase saturation by 30

    # Apply contrast stretching to V channel
    v_channel = sr_hsv[:,:,2].astype(np.float32)  # Convert to float32 for calculations
    min_val = np.percentile(v_channel, 5)
    max_val = np.percentile(v_channel, 95)

    # Güvenli kontrast germe
    if max_val > min_val:
        v_channel = ((v_channel - min_val) * 255.0 / (max_val - min_val))
    else:
        # Eğer max_val = min_val ise, görüntü tamamen düz demektir
        # Bu durumda orijinal değerleri koru
        v_channel = sr_hsv[:,:,2].astype(np.float32)

    # Değerleri [0, 255] aralığına kırp ve uint8'e dönüştür
    v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)
    sr_hsv[:,:,2] = v_channel

    # Convert back to BGR
    sr_bgr = cv2.cvtColor(sr_hsv, cv2.COLOR_HSV2BGR)

    # Create a combined image
    h, w = target_height, target_width
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, :w] = lr_bgr
    combined[:, w:2*w] = sr_bgr
    combined[:, 2*w:] = hr_bgr

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Low Res', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Super Res', (w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'High Res', (2*w+10, 30), font, 1, (255, 255, 255), 2)

    # Save the combined image directly using OpenCV
    filename = f"comparison_{index}.png" if epoch is None else f"epoch_{epoch}_comparison_{index}.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, combined)

def test_model(model, test_loader, device, output_dir):
    """
    Test the model and save comparison images
    """
    model.eval()
    test_psnr = []
    test_ssim = []

    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'test_results')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Clear previous results
    os.makedirs(output_dir, exist_ok=True)

    video_output_dir = os.path.join(output_dir, 'videos')
    os.makedirs(video_output_dir, exist_ok=True)

    # Select a few random videos to create full output videos
    videos_to_save = random.sample(range(len(test_loader)), min(5, len(test_loader)))

    with torch.no_grad():
        for i, (lr_videos, hr_videos) in enumerate(test_loader):
            # Process each frame in the video batch
            batch_size, seq_len, c, h, w = lr_videos.shape

            # Initialize lists to store frames for video creation
            lr_frames = []
            sr_frames = []
            hr_frames = []

            for frame_idx in range(seq_len):
                # Process individual frames
                lr_frame = lr_videos[:, frame_idx].to(device)
                hr_frame = hr_videos[:, frame_idx].to(device)

                # Super-resolve the frame
                sr_frame = model(lr_frame)

                # Calculate metrics
                curr_psnr = calculate_psnr(sr_frame, hr_frame)
                curr_ssim = calculate_ssim(sr_frame, hr_frame)
                test_psnr.append(curr_psnr.item())
                test_ssim.append(curr_ssim.item())

                # Save comparison for the first frame
                if frame_idx == 0:
                    save_comparison(
                        lr_frame[0],
                        sr_frame[0],
                        hr_frame[0],
                        output_dir,
                        i
                    )

                # Store frames for video creation if this video is selected for saving
                if i in videos_to_save:
                    # Convert tensors to numpy for video saving
                    lr_np = (torch.clamp(lr_frame[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    sr_np = (torch.clamp(sr_frame[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    hr_np = (torch.clamp(hr_frame[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    # Get target size from HR image
                    target_height, target_width = hr_np.shape[:2]

                    # Resize LR image to match HR size
                    lr_np = cv2.resize(lr_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

                    # Convert from RGB to BGR for OpenCV
                    lr_np = cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR)
                    sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
                    hr_np = cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR)

                    lr_frames.append(lr_np)
                    sr_frames.append(sr_np)
                    hr_frames.append(hr_np)

            # Create and save videos for selected test samples
            if i in videos_to_save and len(lr_frames) > 0:
                # Get dimensions from the HR frame
                height, width, _ = hr_frames[0].shape

                # Create side-by-side comparison video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(video_output_dir, f'video_comparison_{i}.mp4')

                # Create a writer for the combined video (3x width for LR, SR, HR side by side)
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width*3, height))

                for j in range(len(lr_frames)):
                    # Create a combined frame
                    combined_frame = np.zeros((height, width*3, 3), dtype=np.uint8)
                    combined_frame[:, :width] = lr_frames[j]
                    combined_frame[:, width:2*width] = sr_frames[j]
                    combined_frame[:, 2*width:] = hr_frames[j]

                    # Add frame labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined_frame, 'Low Res', (10, 30), font, 1, (255, 255, 255), 2)
                    cv2.putText(combined_frame, 'Super Res', (width+10, 30), font, 1, (255, 255, 255), 2)
                    cv2.putText(combined_frame, 'High Res', (2*width+10, 30), font, 1, (255, 255, 255), 2)

                    # Write the frame
                    video_writer.write(combined_frame)

                video_writer.release()
                print(f"Saved video comparison to {video_path}")

    # Calculate average metrics
    avg_psnr = sum(test_psnr) / len(test_psnr)
    avg_ssim = sum(test_ssim) / len(test_ssim)

    # Save metrics to a file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Test Results: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim 