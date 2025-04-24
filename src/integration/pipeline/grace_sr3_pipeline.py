#!/usr/bin/env python3
"""
Grace-SR3 Pipeline for Video Processing

This pipeline:
1. Takes a 240p input video
2. Downscales it to 128p
3. Processes with Grace model (trained on 128p)
4. Upscales back to 240p using SR3 model
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from external.Grace.grace.grace_gpu_interface import GraceInterface

def parse_args():
    parser = argparse.ArgumentParser(description='Grace-SR3 Pipeline for Video Processing')
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save output video')
    parser.add_argument('--grace_model', type=str, default='external/Grace/models/grace_128p.pt', help='Path to Grace model')
    parser.add_argument('--sr3_model', type=str, default='external/SR3/models/sr3_128p_to_240p.pt', help='Path to SR3 model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--temp_dir', type=str, default='temp', help='Temporary directory for processing')
    return parser.parse_args()

def load_grace_model(model_path):
    """Load Grace model trained on 128p images"""
    print(f"Loading Grace model from {model_path}")
    config = {
        "path": model_path,
        "device": "cuda"
    }
    model = GraceInterface(config, use_half=True, scale_factor=0.5)
    return model

def load_sr3_model(model_path):
    """Load SR3 model for upscaling from 128p to 240p"""
    print(f"Loading SR3 model from {model_path}")
    model = torch.load(model_path, map_location='cuda')
    model.eval()
    return model

def downscale_frame(frame, target_size=(128, 128)):
    """Downscale frame to 128p"""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def upscale_frame(frame, target_size=(240, 240)):
    """Upscale frame to 240p"""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

def process_batch_with_grace(model, batch, reference_frame):
    """Process batch of frames with Grace model"""
    with torch.no_grad():
        # Convert to torch tensors and normalize
        batch = torch.from_numpy(batch).float().permute(0, 3, 1, 2).cuda() / 255.0
        reference_frame = torch.from_numpy(reference_frame).float().permute(2, 0, 1).cuda() / 255.0
        
        # Process each frame in batch
        processed_frames = []
        for frame in batch:
            # Encode with Grace
            code = model.encode(frame, reference_frame)
            # Decode with Grace
            decoded = model.decode(code, reference_frame)
            processed_frames.append(decoded)
            
            # Update reference frame
            reference_frame = decoded
        
        # Convert back to numpy
        output = torch.stack(processed_frames)
        output = (output.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return output, reference_frame

def process_batch_with_sr3(model, batch):
    """Process batch of frames with SR3 model"""
    with torch.no_grad():
        batch = torch.from_numpy(batch).float().permute(0, 3, 1, 2).cuda() / 255.0
        output = model(batch)
        output = (output.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return output

def process_video(args):
    """Main video processing function"""
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Load models
    grace_model = load_grace_model(args.grace_model)
    sr3_model = load_sr3_model(args.sr3_model)
    
    # Open video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (240, 240))
    
    # Process frames
    batch = []
    frame_count = 0
    reference_frame = None
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Downscale to 128p
            frame_128p = downscale_frame(frame, (128, 128))
            
            # Initialize reference frame if needed
            if reference_frame is None:
                reference_frame = frame_128p
            
            batch.append(frame_128p)
            
            if len(batch) == args.batch_size:
                # Process with Grace
                batch = np.array(batch)
                processed_batch, reference_frame = process_batch_with_grace(grace_model, batch, reference_frame)
                
                # Process with SR3
                upscaled_batch = process_batch_with_sr3(sr3_model, processed_batch)
                
                # Write frames
                for frame in upscaled_batch:
                    out.write(frame)
                
                batch = []
                frame_count += args.batch_size
                pbar.update(args.batch_size)
    
    # Process remaining frames
    if batch:
        batch = np.array(batch)
        processed_batch, _ = process_batch_with_grace(grace_model, batch, reference_frame)
        upscaled_batch = process_batch_with_sr3(sr3_model, processed_batch)
        
        for frame in upscaled_batch:
            out.write(frame)
        
        frame_count += len(batch)
        pbar.update(len(batch))
    
    # Cleanup
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames")
    print(f"Output saved to: {args.output_video}")

def main():
    args = parse_args()
    process_video(args)

if __name__ == "__main__":
    main() 