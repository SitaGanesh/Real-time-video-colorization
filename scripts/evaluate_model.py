# scripts/evaluate_model.py
"""
Model evaluation script
Usage: python scripts/evaluate_model.py --model checkpoints/best_model.pth --data data/processed_frames
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from src.dataset import get_dataloader
from src.inference import RealTimeColorizer
from models.losses import PerceptualLoss

class ModelEvaluator:
    """Evaluate colorization model performance"""
    
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.colorizer = RealTimeColorizer(model_path, self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
    
    def evaluate_dataset(self, data_loader, save_samples=True, num_samples=10):
        """Evaluate model on dataset"""
        self.colorizer.model.eval()
        
        metrics = {
            'mse': [],
            'mae': [],
            'perceptual': [],
            'psnr': [],
            'ssim': []
        }
        
        sample_count = 0
        sample_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                grayscale = batch['grayscale'].to(self.device)
                color_target = batch['color'].to(self.device)
                
                # Ensure grayscale is single channel
                if grayscale.shape[1] == 3:
                    grayscale = torch.mean(grayscale, dim=1, keepdim=True)
                
                # Predict
                predicted = self.colorizer.model(grayscale)
                
                # Compute metrics for each image in batch
                for i in range(grayscale.shape[0]):
                    pred_img = predicted[i].cpu().numpy()
                    target_img = color_target[i].cpu().numpy()
                    
                    # Denormalize
                    pred_img = ((pred_img + 1.0) / 2.0).clip(0, 1)
                    target_img = ((target_img + 1.0) / 2.0).clip(0, 1)
                    
                    # Compute metrics
                    mse = mean_squared_error(target_img.flatten(), pred_img.flatten())
                    mae = mean_absolute_error(target_img.flatten(), pred_img.flatten())
                    psnr = self._compute_psnr(target_img, pred_img)
                    ssim = self._compute_ssim(target_img, pred_img)
                    
                    metrics['mse'].append(mse)
                    metrics['mae'].append(mae)
                    metrics['psnr'].append(psnr)
                    metrics['ssim'].append(ssim)
                
                # Compute perceptual loss for batch
                perceptual = self.perceptual_loss(predicted, color_target).item()
                metrics['perceptual'].extend([perceptual] * grayscale.shape[0])
                
                # Save sample results
                if save_samples and sample_count < num_samples:
                    for i in range(min(grayscale.shape[0], num_samples - sample_count)):
                        gray_np = grayscale[i, 0].cpu().numpy()
                        pred_np = predicted[i].cpu().numpy()
                        target_np = color_target[i].cpu().numpy()
                        
                        sample_results.append({
                            'grayscale': gray_np,
                            'predicted': pred_np,
                            'target': target_np,
                            'filename': batch.get('filename', [f'sample_{sample_count}'])[i]
                        })
                        
                        sample_count += 1
                        if sample_count >= num_samples:
                            break
        
        # Compute average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        return avg_metrics, sample_results
    
    def _compute_psnr(self, img1, img2):
        """Compute Peak Signal-to-Noise Ratio"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def _compute_ssim(self, img1, img2):
        """Compute Structural Similarity Index (simplified version)"""
        # This is a simplified SSIM - for full implementation use skimage
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
        
        return numerator / denominator
    
    def save_sample_results(self, sample_results, output_dir):
        """Save sample results as images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample in enumerate(sample_results):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Grayscale input
            axes[0].imshow(sample['grayscale'], cmap='gray')
            axes[0].set_title('Grayscale Input')
            axes[0].axis('off')
            
            # Predicted colorization
            pred_img = sample['predicted']
            pred_img = ((pred_img + 1.0) / 2.0).clip(0, 1)
            pred_img = np.transpose(pred_img, (1, 2, 0))
            axes[1].imshow(pred_img)
            axes[1].set_title('Predicted')
            axes[1].axis('off')
            
            # Ground truth
            target_img = sample['target']
            target_img = ((target_img + 1.0) / 2.0).clip(0, 1)
            target_img = np.transpose(target_img, (1, 2, 0))
            axes[2].imshow(target_img)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            filename = sample.get('filename', f'sample_{i}')
            plt.suptitle(f'Sample: {filename}')
            plt.tight_layout()
            
            save_path = output_dir / f'sample_{i:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def evaluate_single_image(self, image_path):
        """Evaluate single image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Colorize
        colorized = self.colorizer.process_frame(img_gray)
        
        return {
            'original': img_rgb,
            'grayscale': img_gray,
            'colorized': colorized
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate colorization model')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', help='Path to test data directory')
    parser.add_argument('--image', help='Path to single image for evaluation')
    parser.add_argument('--output', default='evaluation_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sample images to save')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model)
    output_dir = Path(args.output)
    
    if args.image:
        # Evaluate single image
        print(f"Evaluating single image: {args.image}")
        result = evaluator.evaluate_single_image(args.image)
        
        # Save result
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(result['grayscale'], cmap='gray')
        axes[0].set_title('Grayscale Input')
        axes[0].axis('off')
        
        axes[1].imshow(result['colorized'])
        axes[1].set_title('Colorized')
        axes[1].axis('off')
        
        axes[2].imshow(result['original'])
        axes[2].set_title('Original')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'single_image_result.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    elif args.data:
        # Evaluate on dataset
        print(f"Evaluating on dataset: {args.data}")
        
        test_loader = get_dataloader(
            args.data,
            split='validation',
            batch_size=args.batch_size,
            shuffle=False
        )
        
        print(f"Test samples: {len(test_loader.dataset)}")
        
        metrics, sample_results = evaluator.evaluate_dataset(
            test_loader,
            save_samples=True,
            num_samples=args.num_samples
        )
        
        # Print results
        print("\n=== EVALUATION RESULTS ===")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save sample images
        evaluator.save_sample_results(sample_results, output_dir / 'samples')
        
        print(f"\nResults saved to: {output_dir}")
    
    else:
        print("Please specify either --data for dataset evaluation or --image for single image")

if __name__ == '__main__':
    main()
