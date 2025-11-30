"""
Visualization utilities for segmentation
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config


def denormalize_image(image):
    """
    Denormalize image from ImageNet normalization
    
    Args:
        image: (3, H, W) tensor
        
    Returns:
        (H, W, 3) numpy array in [0, 255]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Denormalize
    image = image.transpose(1, 2, 0)  # (H, W, 3)
    image = std * image + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def visualize_prediction(image, gt_mask, pred_mask, prompt, save_path=None):
    """
    Visualize original image, ground truth mask, and predicted mask
    
    Args:
        image: (3, H, W) tensor or (H, W, 3) numpy array
        gt_mask: (H, W) ground truth mask
        pred_mask: (H, W) predicted mask
        prompt: Text prompt
        save_path: Optional path to save the figure
    """
    # Prepare image
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)
    
    # Prepare masks
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    # Remove channel dimension if present
    if len(gt_mask.shape) == 3:
        gt_mask = gt_mask.squeeze(0)
    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask.squeeze(0)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(image)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Add prompt as suptitle
    fig.suptitle(f'Prompt: "{prompt}"', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(samples, save_path=None):
    """
    Create a grid of comparisons (original | GT | prediction)
    
    Args:
        samples: List of dictionaries with keys: image, gt_mask, pred_mask, prompt
        save_path: Optional path to save the figure
    """
    n_samples = len(samples)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        image = sample['image']
        gt_mask = sample['gt_mask']
        pred_mask = sample['pred_mask']
        prompt = sample['prompt']
        
        # Prepare image
        if isinstance(image, torch.Tensor):
            image = denormalize_image(image)
        
        # Prepare masks
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        
        # Remove channel dimension
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask.squeeze(0)
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask.squeeze(0)
        
        # Original
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original\n"{prompt}"')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(gt_mask, alpha=0.5, cmap='jet')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(pred_mask, alpha=0.5, cmap='jet')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train')
    axes[0, 1].plot(history['val_iou'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('IoU Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(history['train_dice'], label='Train')
    axes[1, 0].plot(history['val_dice'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].set_title('Dice Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_prediction_mask(mask, save_path, image_id, prompt):
    """
    Save prediction mask in required format
    
    Args:
        mask: (H, W) mask with values in [0, 1]
        save_path: Directory to save the mask
        image_id: Image ID
        prompt: Text prompt
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Remove channel dimension if present
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)
    
    # Convert to {0, 255}
    mask = (mask * 255).astype(np.uint8)
    
    # Create filename: {image_id}__{prompt}.png
    prompt_clean = prompt.replace(' ', '_').replace('/', '_')
    filename = f"{image_id}__{prompt_clean}.png"
    
    # Save
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    output_path = save_path / filename
    cv2.imwrite(str(output_path), mask)
    
    return output_path


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization...")
    
    # Create dummy data
    image = torch.randn(3, 512, 512)
    gt_mask = torch.randint(0, 2, (512, 512)).float()
    pred_mask = torch.rand(512, 512)
    prompt = "segment crack"
    
    # Test single visualization
    visualize_prediction(image, gt_mask, pred_mask, prompt)
    
    print("\n✓ Visualization test passed!")
