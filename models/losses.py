"""
Loss functions for segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 1, H, W) - logits or probabilities
            targets: (batch_size, 1, H, W) - binary masks {0, 1}
        """
        # Apply sigmoid if predictions are logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # Dice loss is 1 - Dice coefficient
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 1, H, W) - logits
            targets: (batch_size, 1, H, W) - binary masks {0, 1}
        """
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss, {
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'total_loss': total_loss.item()
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, 1, H, W) - logits
            targets: (batch_size, 1, H, W) - binary masks {0, 1}
        """
        # Apply sigmoid
        probs = torch.sigmoid(predictions)
        
        # Compute focal loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Compute focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Final loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()


def get_loss_function(loss_type='combined'):
    """
    Get loss function by name
    
    Args:
        loss_type: 'bce', 'dice', 'combined', or 'focal'
        
    Returns:
        Loss function
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'combined':
        return CombinedLoss()
    elif loss_type == 'focal':
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 4
    predictions = torch.randn(batch_size, 1, 256, 256)
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test BCE loss
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(predictions, targets)
    print(f"BCE Loss: {loss.item():.4f}")
    
    # Test Dice loss
    dice_loss = DiceLoss()
    loss = dice_loss(predictions, targets)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test Combined loss
    combined_loss = CombinedLoss()
    loss, loss_dict = combined_loss(predictions, targets)
    print(f"Combined Loss: {loss.item():.4f}")
    print(f"  BCE: {loss_dict['bce_loss']:.4f}")
    print(f"  Dice: {loss_dict['dice_loss']:.4f}")
    
    # Test Focal loss
    focal_loss = FocalLoss()
    loss = focal_loss(predictions, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    print("\n✓ Loss function tests passed!")
