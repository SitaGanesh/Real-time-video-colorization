# src/inference.py
"""
Real-time inference engine for video colorization
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time

class RealTimeColorizer:
    """Real-time video colorization inference engine"""
    
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        
        print(f"Colorizer initialized on {self.device}")
    
    def _load_model(self, model_path):
        """Load the colorization model"""
        try:
            # Import model classes
            import sys
            sys.path.append('models')
            from colorization_net import ColorizationNet, FastColorizationNet
            
            # Try to determine model type from filename
            if 'fast' in Path(model_path).stem.lower():
                model = FastColorizationNet()
            else:
                model = ColorizationNet()
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            else:
                model.load_state_dict(checkpoint)
            
            return model.to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_frame(self, gray_frame):
        """Preprocess grayscale frame for model input"""
        # Ensure grayscale
        if len(gray_frame.shape) == 3:
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize if needed (model expects certain sizes)
        target_size = (256, 256)  # Adjust based on your model
        if gray_frame.shape[:2] != target_size:
            gray_frame = cv2.resize(gray_frame, target_size)
        
        # Convert to tensor
        tensor = torch.from_numpy(gray_frame).float().unsqueeze(0).unsqueeze(0)
        
        # Normalize to [0, 1]
        tensor = tensor / 255.0
        
        return tensor.to(self.device)
    
    def postprocess_output(self, output, target_shape):
        """Postprocess model output to displayable image"""
        # Convert to numpy
        output = output.squeeze().cpu().detach().numpy()
        
        # Denormalize from [-1, 1] to [0, 255]
        output = ((output + 1.0) / 2.0) * 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Rearrange channels (C, H, W) -> (H, W, C)
        if len(output.shape) == 3:
            output = output.transpose(1, 2, 0)
        
        # Resize to target shape
        if output.shape[:2] != target_shape:
            output = cv2.resize(output, (target_shape[1], target_shape[0]))
        
        return output
    
    def process_frame(self, gray_frame):
        """Process a single grayscale frame"""
        start_time = time.time()
        
        original_shape = gray_frame.shape[:2]
        
        try:
            # Preprocess
            input_tensor = self.preprocess_frame(gray_frame)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocess
            colorized = self.postprocess_output(output, original_shape)
            
            # Update performance metrics
            self.frame_count += 1
            self.total_time += time.time() - start_time
            
            return colorized
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Return grayscale as RGB if processing fails
            if len(gray_frame.shape) == 2:
                return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            return gray_frame
    
    def get_fps(self):
        """Get average processing FPS"""
        if self.total_time > 0:
            return self.frame_count / self.total_time
        return 0
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.total_time = 0

class BatchColorizer:
    """Batch processing for video files"""
    
    def __init__(self, model_path, device=None, batch_size=8):
        self.colorizer = RealTimeColorizer(model_path, device)
        self.batch_size = batch_size
    
    def process_video(self, input_path, output_path, progress_callback=None):
        """Process entire video file"""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Colorize
                colorized = self.colorizer.process_frame(gray)
                
                # Convert BGR for video writer
                colorized_bgr = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(colorized_bgr)
                
                processed_frames += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(processed_frames, total_frames)
        
        finally:
            cap.release()
            out.release()
        
        return processed_frames
