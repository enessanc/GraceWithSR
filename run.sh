#!/bin/bash

# Exit on error
set -e

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check CUDA availability
if ! python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is not available. Please ensure you have a CUDA-capable GPU and drivers installed."
    exit 1
fi

# Check if input video exists
if [ ! -f "dataset/input/input_video.mp4" ]; then
    echo "Input video not found at dataset/input/input_video.mp4"
    exit 1
fi

# Check if models exist
if [ ! -f "models/Grace/grace_128p.pt" ]; then
    echo "Grace model not found at models/Grace/grace_128p.pt"
    exit 1
fi

if [ ! -f "models/SR3/sr3_128p_to_240p.pt" ]; then
    echo "SR3 model not found at models/SR3/sr3_128p_to_240p.pt"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p dataset/output

# Run the pipeline
echo "Starting video processing pipeline..."
python3 src/integration/pipeline/grace_sr3_pipeline.py \
    --input_video dataset/input/input_video.mp4 \
    --output_video dataset/output/output_video.mp4 \
    --grace_model models/Grace/grace_128p.pt \
    --sr3_model models/SR3/sr3_128p_to_240p.pt \
    --batch_size 1

echo "Pipeline completed successfully!"
echo "Output video saved to: dataset/output/output_video.mp4" 