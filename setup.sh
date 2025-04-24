#!/bin/bash

# Exit on error
set -e

echo "Setting up Grace-SR3 Pipeline environment..."

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    unzip

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
pip install -r requirements.txt

# Install Grace dependencies
cd external/Grace
pip install -r requirements.txt
cd ../..

# Install SR3 dependencies
cd external/SR3
pip install -r requirements.txt
cd ../..

# Create necessary directories
mkdir -p models/Grace
mkdir -p models/SR3
mkdir -p dataset/input
mkdir -p dataset/output

# Download Grace models
echo "Downloading Grace models..."
cd models/Grace
wget https://storage.googleapis.com/grace-models/128_freeze.model -O grace_128p.pt
cd ../..

# Download SR3 models
echo "Downloading SR3 models..."
cd models/SR3
wget https://storage.googleapis.com/sr3-models/128_to_240.pt -O sr3_128p_to_240p.pt
cd ../..

echo "Setup completed successfully!"
echo "To activate the environment, run: source venv/bin/activate" 