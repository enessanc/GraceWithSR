# Grace-SR3 Video Processing Pipeline

This pipeline processes videos by first downscaling them to 128p, applying the Grace model for enhancement, and then upscaling back to 240p using the SR3 model.

## Workflow

1. **Input**: 240p video
2. **Downscaling**: Video is downscaled to 128p resolution
3. **Grace Processing**: Each frame is processed by the Grace model (trained on 128p images)
4. **SR3 Upscaling**: Processed frames are upscaled back to 240p using the SR3 model
5. **Output**: Final enhanced video at 240p resolution

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- OpenCV
- NumPy
- tqdm

## Usage

```bash
python src/integration/pipeline/grace_sr3_pipeline.py \
    --input_video path/to/input/video.mp4 \
    --output_video path/to/output/video.mp4 \
    --grace_model path/to/grace_128p.pt \
    --sr3_model path/to/sr3_128p_to_240p.pt \
    --batch_size 1
```

## Arguments

- `--input_video`: Path to the input video file (required)
- `--output_video`: Path to save the output video (required)
- `--grace_model`: Path to Grace model trained on 128p images (default: external/Grace/models/grace_128p.pt)
- `--sr3_model`: Path to SR3 model for upscaling (default: external/SR3/models/sr3_128p_to_240p.pt)
- `--batch_size`: Number of frames to process at once (default: 1)
- `--temp_dir`: Directory for temporary files (default: temp)

## Notes

- The pipeline requires CUDA-enabled GPU for optimal performance
- Input video should be 240p resolution
- Grace model should be trained on 128p images
- SR3 model should be trained for upscaling from 128p to 240p
