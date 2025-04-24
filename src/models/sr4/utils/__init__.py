import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import gc
import shutil

from .trainer import setup_cuda, print_memory_status, train_epoch, validate, train_with_validation
from .tester import calculate_psnr, calculate_ssim, save_comparison, test_model

__all__ = [
    'setup_cuda', 'print_memory_status', 'train_epoch', 'validate', 'train_with_validation',
    'calculate_psnr', 'calculate_ssim', 'save_comparison', 'test_model'
]

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
