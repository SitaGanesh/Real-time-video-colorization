# scripts/prepare_data.py
"""
Data preparation script for video colorization
Usage: python scripts/prepare_data.py --input data/raw_videos --output data/processed_frames
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import shutil
from multiprocessing import Pool
import functools

def process_video(args):
    """Process a single video file"""
    video_path, output_dir, frame_rate, max_frames, target_size = args
    
    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / frame_rate))
    
    frame_count = 0
    saved_count = 0
    
    # Create directories
    train_gray_dir = output_dir / 'train' / 'grayscale'
    train_color_dir = output_dir / 'train' / 'color'
    val_gray_dir = output_dir / 'validation' / 'grayscale'
    val_color_dir = output_dir / 'validation' / 'color'
    
    for dir_path in [train_gray_dir, train_color_dir, val_gray_dir, val_color_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Resize frame
                if target_size:
                    frame = cv2.resize(frame, target_size)
                
                # Create grayscale version
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Decide train/validation split (80/20)
                is_train = np.random.random() > 0.2
                
                if is_train:
                    color_path = train_color_dir / f'{video_name}_{saved_count:06d}.jpg'
                    gray_path = train_gray_dir / f'{video_name}_{saved_count:06d}.jpg'
                else:
                    color_path = val_color_dir / f'{video_name}_{saved_count:06d}.jpg'
                    gray_path = val_gray_dir / f'{video_name}_{saved_count:06d}.jpg'
                
                # Save frames
                cv2.imwrite(str(color_path), frame)
                cv2.imwrite(str(gray_path), gray_frame)
                
                saved_count += 1
            
            frame_count += 1
    
    finally:
        cap.release()
    
    return saved_count

def main():
    parser = argparse.ArgumentParser(description='Prepare video data for training')
    parser.add_argument('--input', '-i', required=True, help='Input directory with videos')
    parser.add_argument('--output', '-o', required=True, help='Output directory for processed frames')
    parser.add_argument('--frame-rate', type=float, default=1.0, help='Frames per second to extract')
    parser.add_argument('--max-frames', type=int, default=500, help='Max frames per video')
    parser.add_argument('--target-size', type=str, help='Target size (WxH), e.g., 256x256')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Parse target size
    target_size = None
    if args.target_size:
        try:
            w, h = map(int, args.target_size.split('x'))
            target_size = (w, h)
        except ValueError:
            print(f"Invalid target size format: {args.target_size}")
            return
    
    # Find video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f'*{ext}')))
        video_files.extend(list(input_dir.glob(f'*{ext.upper()}')))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    print(f"Extracting {args.frame_rate} fps, max {args.max_frames} frames per video")
    if target_size:
        print(f"Target size: {target_size}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    process_args = [
        (video_path, output_dir, args.frame_rate, args.max_frames, target_size)
        for video_path in video_files
    ]
    
    # Process videos in parallel
    total_frames = 0
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_video, process_args),
            total=len(video_files),
            desc="Processing videos"
        ))
    
    total_frames = sum(results)
    print(f"\nProcessed {len(video_files)} videos")
    print(f"Extracted {total_frames} total frames")
    
    # Print statistics
    train_gray_dir = output_dir / 'train' / 'grayscale'
    train_color_dir = output_dir / 'train' / 'color'
    val_gray_dir = output_dir / 'validation' / 'grayscale'
    val_color_dir = output_dir / 'validation' / 'color'
    
    train_count = len(list(train_gray_dir.glob('*.jpg')))
    val_count = len(list(val_gray_dir.glob('*.jpg')))
    
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Data saved to: {output_dir}")

if __name__ == '__main__':
    main()
