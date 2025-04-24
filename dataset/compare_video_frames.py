import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def extract_frame(video_path, frame_number=10):
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_number >= total_frames:
        raise ValueError(f"Frame number {frame_number} is out of range. Video has {total_frames} frames.")
    
    # Set to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_number}")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame

def resize_frames(frame1, frame2):
    """Resize frames to have exactly the same dimensions."""
    # Get dimensions of both frames
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Calculate target dimensions (use the smaller values)
    target_width = min(w1, w2)
    target_height = min(h1, h2)
    
    # Resize both frames to the same dimensions
    resized_frame1 = cv2.resize(frame1, (target_width, target_height))
    resized_frame2 = cv2.resize(frame2, (target_width, target_height))
    
    return resized_frame1, resized_frame2

def display_frames(input_frame, output_frame):
    """Display input and output frames side by side."""
    # Resize frames to have the same dimensions
    input_frame, output_frame = resize_frames(input_frame, output_frame)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(input_frame)
    axes[0].set_title('Input Frame 10')
    axes[0].axis('off')
    
    axes[1].imshow(output_frame)
    axes[1].set_title('Output Frame 10')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare the 10th frame from input and output videos')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to output video file')
    
    args = parser.parse_args()
    
    try:
        input_frame = extract_frame(args.input_video, 10)
        output_frame = extract_frame(args.output_video, 10)
        display_frames(input_frame, output_frame)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 