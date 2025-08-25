# src/dataset.py
"""
Dataset classes for video colorization training
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from torchvision import transforms

class ColorizationDataset(Dataset):
    """Dataset for colorization training"""
    
    def __init__(self, data_dir, split='train', transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augment = augment
        
        # Get image paths
        self.grayscale_dir = self.data_dir / split / 'grayscale'
        self.color_dir = self.data_dir / split / 'color'
        
        self.image_files = self._get_image_files()
        
        # Default transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        
        print(f"Loaded {len(self.image_files)} {split} samples")
    
    def _get_image_files(self):
        """Get list of image files"""
        if not self.grayscale_dir.exists():
            raise ValueError(f"Grayscale directory not found: {self.grayscale_dir}")
        
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            files.extend(list(self.grayscale_dir.glob(ext)))
        
        # Filter files that have corresponding color images
        valid_files = []
        for gray_file in files:
            color_file = self.color_dir / gray_file.name
            if color_file.exists():
                valid_files.append(gray_file.name)
        
        return sorted(valid_files)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load images
        gray_path = self.grayscale_dir / img_name
        color_path = self.color_dir / img_name
        
        gray_img = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        if gray_img is None or color_img is None:
            raise ValueError(f"Failed to load images: {gray_path}, {color_path}")
        
        # Apply augmentations if training
        if self.augment and self.split == 'train':
            gray_img, color_img = self._apply_augmentations(gray_img, color_img)
        
        # Apply transforms
        gray_tensor = self.transform(gray_img)
        color_tensor = self.transform(color_img)
        
        # Ensure grayscale is single channel
        if gray_tensor.shape[0] == 3:
            gray_tensor = gray_tensor[0:1]  # Take first channel
        
        return {
            'grayscale': gray_tensor,
            'color': color_tensor,
            'filename': img_name
        }
    
    def _apply_augmentations(self, gray_img, color_img):
        """Apply data augmentations"""
        # Random horizontal flip
        if random.random() > 0.5:
            gray_img = cv2.flip(gray_img, 1)
            color_img = cv2.flip(color_img, 1)
        
        # Random rotation
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            center = (gray_img.shape[1]//2, gray_img.shape[0]//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray_img = cv2.warpAffine(gray_img, matrix, 
                                    (gray_img.shape[1], gray_img.shape[0]))
            color_img = cv2.warpAffine(color_img, matrix, 
                                     (color_img.shape[1], color_img.shape[0]))
        
        # Random brightness/contrast for color image only
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)    # Brightness
            color_img = cv2.convertScaleAbs(color_img, alpha=alpha, beta=beta)
        
        return gray_img, color_img

class VideoFrameDataset(Dataset):
    """Dataset that extracts frames from videos"""
    
    def __init__(self, video_dir, frame_interval=30, max_frames_per_video=100):
        self.video_dir = Path(video_dir)
        self.frame_interval = frame_interval
        self.max_frames_per_video = max_frames_per_video
        
        self.video_files = self._get_video_files()
        self.frames = self._extract_frames()
    
    def _get_video_files(self):
        """Get list of video files"""
        files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            files.extend(list(self.video_dir.glob(ext)))
        return files
    
    def _extract_frames(self):
        """Extract frames from videos"""
        frames = []
        
        for video_file in self.video_files:
            cap = cv2.VideoCapture(str(video_file))
            
            if not cap.isOpened():
                continue
            
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < self.max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    frames.append(frame)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
        
        return frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensors
        gray_tensor = torch.from_numpy(gray).float().unsqueeze(0) / 255.0
        color_tensor = torch.from_numpy(color).float().permute(2, 0, 1) / 255.0
        
        return {
            'grayscale': gray_tensor,
            'color': color_tensor
        }

def get_dataloader(data_dir, split='train', batch_size=16, num_workers=4, 
                  shuffle=None):
    """Get dataloader for training/validation"""
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = ColorizationDataset(data_dir, split=split, augment=(split=='train'))
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader
