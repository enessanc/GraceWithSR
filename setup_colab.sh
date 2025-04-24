#!/bin/bash

# Exit on error
set -e

echo "Setting up Grace-SR3 Pipeline environment for Google Colab..."

# Install PyTorch with CUDA support
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu118

# Install Grace core dependencies
pip install \
    numpy==1.24.1 \
    scipy>=1.0.1 \
    scikit-image==0.19.3 \
    opencv-python==4.5.5.64 \
    matplotlib==3.6.3 \
    torchac==0.9.3 \
    pillow==9.4.0 \
    pytorch-msssim==0.2.1 \
    range-coder==1.1 \
    compressai==1.2.3 \
    ffmpeg-python==0.2.0 \
    motion-vector-extractor==1.0.6

# Install SR3 core dependencies
pip install \
    numpy>=1.14.3 \
    scipy>=1.0.1 \
    scikit-image>=0.13.0 \
    opencv-python>=2.4.11 \
    matplotlib>=1.5.1 \
    pillow>=9.4.0 \
    pytorch-msssim==0.2.1 \
    einops \
    lpips \
    albumentations \
    opencv-python-headless

# Install additional dependencies
pip install \
    tensorboard==2.11.2 \
    tensorboardx==2.5.1 \
    pandas==1.5.3 \
    pyyaml==6.0 \
    tqdm \
    gdown==4.6.4


echo "Colab setup completed successfully!" 