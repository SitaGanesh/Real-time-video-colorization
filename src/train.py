# src/train.py
"""
Training script for video colorization models
Usage: python src/train.py --config config/training_config.py
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from pathlib import Path
import time

# Add project paths
sys.path.append('.')
sys.path.append('models')
sys.path.append('config')

from src.dataset import ColorizationDataset, get_dataloader
from models.colorization_net import ColorizationNet, FastColorizationNet, AttentionColorizationNet
from models.losses import ColorizationLoss, PerceptualLoss
from config.training_config import TrainingConfig

class Trainer:
    """Training class for colorization models"""
    
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.DEVICE)
        
        # Create directories
        Path(self.config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.config.LOG_DIR).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup loss function
        self.criterion = ColorizationLoss(
            lambda_l1=self.config.LAMBDA_L1,
            lambda_perceptual=self.config.LAMBDA_PERCEPTUAL
        )
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.config.SCHEDULER_FACTOR,
            patience=self.config.SCHEDULER_PATIENCE,
            min_lr=self.config.SCHEDULER_MIN_LR,
            verbose=True
        )
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.config.LOG_DIR)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        print(f"Trainer initialized on {self.device}")
        print(f"Model: {self.config.MODEL_NAME}")
        print(f"Parameters: {self.count_parameters():,}")
    
    def _create_model(self):
        """Create model based on config"""
        if self.config.MODEL_NAME == 'FastColorizationNet':
            model = FastColorizationNet(
                input_channels=self.config.INPUT_CHANNELS,
                output_channels=self.config.OUTPUT_CHANNELS
            )
        elif self.config.MODEL_NAME == 'AttentionColorizationNet':
            model = AttentionColorizationNet(
                input_channels=self.config.INPUT_CHANNELS,
                output_channels=self.config.OUTPUT_CHANNELS
            )
        else:
            model = ColorizationNet(
                input_channels=self.config.INPUT_CHANNELS,
                output_channels=self.config.OUTPUT_CHANNELS
            )
        
        return model.to(self.device)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_perceptual_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            grayscale = batch['grayscale'].to(self.device)
            color = batch['color'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Ensure grayscale is single channel
            if grayscale.shape[1] == 3:
                grayscale = torch.mean(grayscale, dim=1, keepdim=True)
            
            predicted = self.model(grayscale)
            
            # Compute loss
            loss_dict = self.criterion(predicted, color)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss_dict['total_loss'].item()
            epoch_l1_loss += loss_dict['l1_loss'].item()
            epoch_perceptual_loss += loss_dict['perceptual_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'L1': f"{loss_dict['l1_loss'].item():.4f}",
                'Perceptual': f"{loss_dict['perceptual_loss'].item():.4f}"
            })
            
            # Log to tensorboard
            step = self.current_epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss_dict['total_loss'].item(), step)
        
        # Average losses
        epoch_loss /= len(train_loader)
        epoch_l1_loss /= len(train_loader)
        epoch_perceptual_loss /= len(train_loader)
        
        return {
            'total_loss': epoch_loss,
            'l1_loss': epoch_l1_loss,
            'perceptual_loss': epoch_perceptual_loss
        }
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_perceptual_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                grayscale = batch['grayscale'].to(self.device)
                color = batch['color'].to(self.device)
                
                # Ensure grayscale is single channel
                if grayscale.shape[1] == 3:
                    grayscale = torch.mean(grayscale, dim=1, keepdim=True)
                
                predicted = self.model(grayscale)
                
                loss_dict = self.criterion(predicted, color)
                
                val_loss += loss_dict['total_loss'].item()
                val_l1_loss += loss_dict['l1_loss'].item()
                val_perceptual_loss += loss_dict['perceptual_loss'].item()
        
        # Average losses
        val_loss /= len(val_loader)
        val_l1_loss /= len(val_loader)
        val_perceptual_loss /= len(val_loader)
        
        return {
            'total_loss': val_loss,
            'l1_loss': val_l1_loss,
            'perceptual_loss': val_perceptual_loss
        }
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        if epoch % self.config.SAVE_EVERY == 0:
            checkpoint_path = Path(self.config.CHECKPOINT_DIR) / f'model_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = Path(self.config.CHECKPOINT_DIR) / 'model_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.CHECKPOINT_DIR) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss: {loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from=None):
        """Main training loop"""
        # Load checkpoint if resuming
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Get data loaders
        train_loader = get_dataloader(
            self.config.DATA_DIR,
            split='train',
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS
        )
        
        val_loader = get_dataloader(
            self.config.DATA_DIR,
            split='validation',
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Log training metrics
            self.writer.add_scalar('Train/EpochLoss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Train/L1Loss', train_metrics['l1_loss'], epoch)
            self.writer.add_scalar('Train/PerceptualLoss', train_metrics['perceptual_loss'], epoch)
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Validate
            if epoch % self.config.VALIDATE_EVERY == 0:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                self.writer.add_scalar('Val/EpochLoss', val_metrics['total_loss'], epoch)
                self.writer.add_scalar('Val/L1Loss', val_metrics['l1_loss'], epoch)
                self.writer.add_scalar('Val/PerceptualLoss', val_metrics['perceptual_loss'], epoch)
                
                # Check for improvement
                is_best = val_metrics['total_loss'] < self.best_loss
                if is_best:
                    self.best_loss = val_metrics['total_loss']
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics['total_loss'], is_best)
                
                # Update scheduler
                self.scheduler.step(val_metrics['total_loss'])
                
                print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
                print(f"Train Loss: {train_metrics['total_loss']:.4f}")
                print(f"Val Loss: {val_metrics['total_loss']:.4f}")
                print(f"Best Loss: {self.best_loss:.4f}")
                print("-" * 50)
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            else:
                # Save checkpoint even if not validating
                self.save_checkpoint(epoch, train_metrics['total_loss'])
        
        self.writer.close()
        print("Training completed!")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train video colorization model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--model', type=str, choices=['ColorizationNet', 'FastColorizationNet', 'AttentionColorizationNet'],
                       help='Model architecture')
    
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig()
    
    # Override config with command line arguments
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.model:
        config.MODEL_NAME = args.model
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train(resume_from=args.resume)

if __name__ == '__main__':
    main()
