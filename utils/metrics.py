"""
Evaluation metrics for segmentation
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_iou(predictions, targets, threshold=0.5):
    """
    Compute Intersection over Union (IoU)
    
    Args:
        predictions: (N, H, W) - predicted probabilities or logits
        targets: (N, H, W) - ground truth binary masks {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        IoU score
    """
    # Binarize predictions
    if predictions.max() > 1.0:
        predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Compute intersection and union
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def compute_dice(predictions, targets, threshold=0.5):
    """
    Compute Dice coefficient
    
    Args:
        predictions: (N, H, W) - predicted probabilities or logits
        targets: (N, H, W) - ground truth binary masks {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dice score
    """
    # Binarize predictions
    if predictions.max() > 1.0:
        predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Compute Dice coefficient
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection) / (predictions.sum() + targets.sum())
    
    # Handle edge case
    if predictions.sum() == 0 and targets.sum() == 0:
        return 1.0
    
    return dice.item()


def compute_precision_recall_f1(predictions, targets, threshold=0.5):
    """
    Compute precision, recall, and F1 score
    
    Args:
        predictions: (N, H, W) - predicted probabilities or logits
        targets: (N, H, W) - ground truth binary masks {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dictionary with precision, recall, and f1
    """
    # Binarize predictions
    if predictions.max() > 1.0:
        predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    # Flatten
    predictions = predictions.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class MetricsTracker:
    """
    Track metrics during training and evaluation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.iou_scores = []
        self.dice_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
    
    def update(self, predictions, targets, threshold=0.5):
        """
        Update metrics with a batch of predictions
        
        Args:
            predictions: (batch_size, 1, H, W) - predicted logits or probabilities
            targets: (batch_size, 1, H, W) - ground truth binary masks
            threshold: Threshold for binarizing predictions
        """
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            # Compute metrics
            iou = compute_iou(pred, target, threshold)
            dice = compute_dice(pred, target, threshold)
            prf = compute_precision_recall_f1(pred, target, threshold)
            
            # Store
            self.iou_scores.append(iou)
            self.dice_scores.append(dice)
            self.precision_scores.append(prf['precision'])
            self.recall_scores.append(prf['recall'])
            self.f1_scores.append(prf['f1'])
    
    def get_metrics(self):
        """
        Get average metrics
        
        Returns:
            Dictionary with average metrics
        """
        return {
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'precision': np.mean(self.precision_scores) if self.precision_scores else 0.0,
            'recall': np.mean(self.recall_scores) if self.recall_scores else 0.0,
            'f1': np.mean(self.f1_scores) if self.f1_scores else 0.0
        }
    
    def get_std(self):
        """
        Get standard deviation of metrics
        
        Returns:
            Dictionary with std of metrics
        """
        return {
            'iou_std': np.std(self.iou_scores) if self.iou_scores else 0.0,
            'dice_std': np.std(self.dice_scores) if self.dice_scores else 0.0,
            'precision_std': np.std(self.precision_scores) if self.precision_scores else 0.0,
            'recall_std': np.std(self.recall_scores) if self.recall_scores else 0.0,
            'f1_std': np.std(self.f1_scores) if self.f1_scores else 0.0
        }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy data
    predictions = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    # Test individual metrics
    iou = compute_iou(predictions, targets)
    print(f"IoU: {iou:.4f}")
    
    dice = compute_dice(predictions, targets)
    print(f"Dice: {dice:.4f}")
    
    prf = compute_precision_recall_f1(predictions, targets)
    print(f"Precision: {prf['precision']:.4f}")
    print(f"Recall: {prf['recall']:.4f}")
    print(f"F1: {prf['f1']:.4f}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(predictions, targets)
    metrics = tracker.get_metrics()
    
    print("\nMetrics Tracker:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n✓ Metrics tests passed!")
