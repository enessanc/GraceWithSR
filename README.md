# GraceWithSR: Video Enhancement Pipeline

This project implements a video enhancement pipeline that combines Grace and SR3 models to improve video quality. The pipeline:
1. Takes a 240p input video
2. Downscales it to 128p
3. Processes with Grace model (trained on 128p)
4. Upscales back to 240p using SR3 model

## Requirements

- Python 3.7+
- CUDA-capable GPU
- FFmpeg
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/enessanc/GraceWithSR.git
cd GraceWithSR
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install system dependencies
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install Grace and SR3 dependencies
- Download required models

## Usage

1. Place your input video in `dataset/input/input_video.mp4`

2. Run the pipeline:
```bash
chmod +x run.sh
./run.sh
```

The processed video will be saved to `dataset/output/output_video.mp4`

## Project Structure

```
GraceWithSR/
├── src/
│   └── integration/
│       └── pipeline/
│           └── grace_sr3_pipeline.py
├── external/
│   ├── Grace/
│   └── SR3/
├── models/
│   ├── Grace/
│   └── SR3/
├── dataset/
│   ├── input/
│   └── output/
├── setup.sh
├── run.sh
└── requirements.txt
```

## Notes

- The pipeline requires CUDA for optimal performance
- Input video should be 240p resolution
- Grace model is trained on 128p images
- SR3 model is trained for upscaling from 128p to 240p

## License

This project is licensed under the MIT License - see the LICENSE file for details. 