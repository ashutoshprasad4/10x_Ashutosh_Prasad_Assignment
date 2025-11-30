"""
Training script for text-conditioned segmentation
"""
import os
import sys
import time
import json
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from data.dataset import get_dataloaders
from models.clip_segmentation import build_model
from models.losses import CombinedLoss
from utils.metrics import MetricsTracker


class Trainer:
    """Trainer class for segmentation model"""
    
    def __init__(self, args):
        self.args = args
        
        # Set random seed
        config.set_seed(config.RANDOM_SEED)
        
        # Create directories
        config.create_directories()
        
        # Setup device
        self.device = torch.device(config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Build model
        print("Building model...")
        self.model = build_model()
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss function
        self.criterion = CombinedLoss(
            bce_weight=config.BCE_WEIGHT,
            dice_weight=config.DICE_WEIGHT
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.LR_MIN,
            verbose=True
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # Data loaders
        print("Loading datasets...")
        dataloaders = get_dataloaders(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_iou = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Training time tracking
        self.total_train_time = 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        metrics_tracker = MetricsTracker()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            prompts = batch['prompt']
            
            # Forward pass
            if config.USE_AMP:
                with autocast():
                    logits = self.model(images, prompts)
                    loss, loss_dict = self.criterion(logits, masks)
            else:
                logits = self.model(images, prompts)
                loss, loss_dict = self.criterion(logits, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if config.USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_MAX_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_MAX_NORM)
                self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                metrics_tracker.update(logits, masks, threshold=config.PREDICTION_THRESHOLD)
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bce': f"{loss_dict['bce_loss']:.4f}",
                'dice': f"{loss_dict['dice_loss']:.4f}"
            })
            
            # Log to TensorBoard
            if batch_idx % config.LOG_INTERVAL == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = metrics_tracker.get_metrics()
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        metrics_tracker = MetricsTracker()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            prompts = batch['prompt']
            
            # Forward pass
            if config.USE_AMP:
                with autocast():
                    logits = self.model(images, prompts)
                    loss, loss_dict = self.criterion(logits, masks)
            else:
                logits = self.model(images, prompts)
                loss, loss_dict = self.criterion(logits, masks)
            
            # Update metrics
            metrics_tracker.update(logits, masks, threshold=config.PREDICTION_THRESHOLD)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = metrics_tracker.get_metrics()
        
        return avg_loss, metrics
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(config.NUM_EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            self.total_train_time += epoch_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log to TensorBoard
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            self.writer.add_scalar('Epoch/TrainIoU', train_metrics['iou'], epoch)
            self.writer.add_scalar('Epoch/ValIoU', val_metrics['iou'], epoch)
            self.writer.add_scalar('Epoch/TrainDice', train_metrics['dice'], epoch)
            self.writer.add_scalar('Epoch/ValDice', val_metrics['dice'], epoch)
            self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Epoch time: {epoch_time:.2f}s")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['iou'])
            
            # Save best model
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                checkpoint_path = config.CHECKPOINTS_DIR / "best_model.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"  ✓ New best model saved! (IoU: {self.best_val_iou:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % config.SAVE_CHECKPOINT_INTERVAL == 0:
                checkpoint_path = config.CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
            
            # Early stopping
            if self.epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        
        # Save final checkpoint
        final_checkpoint = config.CHECKPOINTS_DIR / "final_model.pth"
        self.save_checkpoint(final_checkpoint, epoch, val_metrics)
        
        # Save training history
        history_path = config.LOGS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Average time per epoch: {self.total_train_time/len(self.history['train_loss']):.2f}s")
        print(f"Best validation IoU: {self.best_val_iou:.4f} (epoch {self.best_epoch+1})")
        print(f"Best model saved at: {config.CHECKPOINTS_DIR / 'best_model.pth'}")
        print("="*60)
        
        self.writer.close()
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': {
                'IMAGE_SIZE': config.IMAGE_SIZE,
                'CLIP_MODEL_NAME': config.CLIP_MODEL_NAME,
                'ENCODER_NAME': config.ENCODER_NAME,
                'RANDOM_SEED': config.RANDOM_SEED
            }
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='Train text-conditioned segmentation model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Create trainer and train
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
