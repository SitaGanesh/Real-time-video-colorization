# scripts/download_pretrained.py
"""
Download pretrained models and sample data
Usage: python scripts/download_pretrained.py
"""

import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import os

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def extract_archive(archive_path, dest_dir):
    """Extract archive file"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(dest_dir)

def main():
    # URLs for pretrained models (these are example URLs - replace with actual)
    models = {
        'fast_model.pth': 'https://example.com/models/fast_model.pth',
        'hq_model.pth': 'https://example.com/models/hq_model.pth',
        'gan_model.pth': 'https://example.com/models/gan_model.pth'
    }
    
    # Sample video URLs (replace with actual URLs)
    sample_videos = {
        'sample_1.mp4': 'https://example.com/videos/sample_1.mp4',
        'sample_2.mp4': 'https://example.com/videos/sample_2.mp4',
        'sample_3.mp4': 'https://example.com/videos/sample_3.mp4',
        'sample_4.mp4': 'https://example.com/videos/sample_4.mp4',
        'sample_5.mp4': 'https://example.com/videos/sample_5.mp4',
        'sample_6.mp4': 'https://example.com/videos/sample_6.mp4'
    }
    
    base_dir = Path('.')
    
    # Download pretrained models
    print("Downloading pretrained models...")
    models_dir = base_dir / 'checkpoints'
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            print(f"Model {model_name} already exists, skipping...")
            continue
        
        try:
            print(f"Downloading {model_name}...")
            download_file(url, model_path)
            print(f"✓ Downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
    
    # Download sample videos
    print("\nDownloading sample videos...")
    videos_dir = base_dir / 'data' / 'raw_videos'
    
    for video_name, url in sample_videos.items():
        video_path = videos_dir / video_name
        
        if video_path.exists():
            print(f"Video {video_name} already exists, skipping...")
            continue
        
        try:
            print(f"Downloading {video_name}...")
            download_file(url, video_path)
            print(f"✓ Downloaded {video_name}")
        except Exception as e:
            print(f"✗ Failed to download {video_name}: {e}")
    
    # Create placeholder files if downloads failed
    print("\nCreating placeholder files...")
    
    # Create basic model files (random weights for testing)
    import torch
    from models.colorization_net import ColorizationNet, FastColorizationNet
    
    for model_name in models.keys():
        model_path = models_dir / model_name
        
        if not model_path.exists():
            print(f"Creating placeholder {model_name}...")
            
            if 'fast' in model_name:
                model = FastColorizationNet()
            else:
                model = ColorizationNet()
            
            # Save with proper format
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 0,
                'loss': 1.0,
                'config': {}
            }
            
            torch.save(checkpoint, model_path)
            print(f"✓ Created placeholder {model_name}")
    
    print("\nSetup complete!")
    print("Note: Replace placeholder models with actual pretrained weights")
    print("Add your training videos to data/raw_videos/")

if __name__ == '__main__':
    main()
