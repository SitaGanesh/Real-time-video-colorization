# src/video_processor.py
"""
Video processing utilities
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import os

class VideoProcessor:
    """Utility class for video processing operations"""
    
    @staticmethod
    def extract_frames(video_path, output_dir, frame_rate=1, max_frames=None):
        """Extract frames from video at specified rate"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate)  # Extract every N frames
        
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Save color frame
                    color_path = output_dir / 'color' / f'{video_path.stem}_{saved_count:06d}.jpg'
                    color_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(color_path), frame)
                    
                    # Save grayscale frame
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_path = output_dir / 'grayscale' / f'{video_path.stem}_{saved_count:06d}.jpg'
                    gray_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(gray_path), gray_frame)
                    
                    saved_count += 1
                    
                    if max_frames and saved_count >= max_frames:
                        break
                
                frame_count += 1
        
        finally:
            cap.release()
        
        return saved_count
    
    @staticmethod
    def create_video_from_frames(frames_dir, output_path, fps=30):
        """Create video from sequence of frames"""
        frames_dir = Path(frames_dir)
        
        # Get frame files
        frame_files = sorted(list(frames_dir.glob('*.jpg')) + 
                           list(frames_dir.glob('*.png')))
        
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                out.write(frame)
        
        finally:
            out.release()
    
    @staticmethod
    def get_video_info(video_path):
        """Get video information"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    @staticmethod
    def resize_video(input_path, output_path, target_size=(256, 256)):
        """Resize video to target dimensions"""
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, target_size)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                resized = cv2.resize(frame, target_size)
                out.write(resized)
        
        finally:
            cap.release()
            out.release()
    
    @staticmethod
    def convert_to_grayscale_video(input_path, output_path):
        """Convert color video to grayscale"""
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), False)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out.write(gray)
        
        finally:
            cap.release()
            out.release()
