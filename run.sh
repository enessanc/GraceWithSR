#!/bin/bash

# Exit on error
set -e

# Check if running in Colab
if [ -d "/content" ]; then
    echo "Running in Google Colab environment..."
    IS_COLAB=true
else
    echo "Running in local environment..."
    IS_COLAB=false
fi

# Check if virtual environment exists and activate it
if [ "$IS_COLAB" = false ] && [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if CUDA is available
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is available. Using GPU..."
    USE_GPU=true
else
    echo "CUDA is not available. Using CPU..."
    USE_GPU=false
fi

# Check if required directories exist
if [ ! -d "external/Grace/models" ]; then
    echo "Error: Grace models directory not found!"
    exit 1
fi

if [ ! -d "models/SR3" ]; then
    echo "Error: SR3 models directory not found!"
    exit 1
fi

# Check if model files exist
if [ ! -f "external/Grace/models/128_freeze.model" ]; then
    echo "Error: Grace model file (128_freeze.model) not found!"
    exit 1
fi

if [ ! -f "models/SR3Model/SR3Model.pth" ]; then
    echo "Error: SR3 model file (SR3Model.pth) not found!"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p dataset/input
mkdir -p dataset/output

# Run the pipeline
echo "Starting Grace-SR3 Pipeline..."

# Set GPU flag for the Python script
if [ "$USE_GPU" = true ]; then
    GPU_FLAG="--use_gpu"
else
    GPU_FLAG=""
fi

# Run the main pipeline script
python3 src/integration/pipeline/grace_sr3_pipeline.py \
    --input dataset/input_video.mp4
    --output dataset/output_video.mp4
    --grace_model external/Grace/models/128_freeze.model \
    --sr3_model models/SR3Model/SR3Model.pth \
    $GPU_FLAG

echo "Pipeline completed successfully!" 