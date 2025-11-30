"""
Evaluation script for text-conditioned segmentation
"""
import os
import sys
import time
import json
from pathlib import Path
import argparse
from collections import defaultdict

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from data.dataset import get_dataloaders
from models.clip_segmentation import build_model
from utils.metrics import MetricsTracker
from utils.visualization import save_prediction_mask


class Evaluator:
    """Evaluator class for segmentation model"""
    
    def __init__(self, checkpoint_path, output_dir=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir) if output_dir else config.PREDICTIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device(config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {self.checkpoint_path}...")
        self.model = build_model()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from epoch {checkpoint['epoch']+1}")
        
        # Get data loaders
        print("Loading datasets...")
        dataloaders = get_dataloaders()
        self.test_loader = dataloaders['test']
        
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    @torch.no_grad()
    def evaluate(self, save_predictions=True):
        """
        Evaluate model on test set
        
        Args:
            save_predictions: Whether to save prediction masks
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*60)
        print("Evaluating model...")
        print("="*60)
        
        # Overall metrics
        overall_tracker = MetricsTracker()
        
        # Per-prompt metrics
        prompt_trackers = defaultdict(MetricsTracker)
        
        # Inference time tracking
        inference_times = []
        
        # Predictions for visualization
        predictions_for_viz = defaultdict(list)
        
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            prompts = batch['prompt']
            image_ids = batch['image_id']
            dataset_types = batch['dataset_type']
            
            # Measure inference time
            start_time = time.time()
            
            if config.USE_AMP:
                with autocast():
                    logits = self.model(images, prompts)
            else:
                logits = self.model(images, prompts)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(images))  # Per image
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs > config.PREDICTION_THRESHOLD).float()
            
            # Update overall metrics
            overall_tracker.update(logits, masks)
            
            # Update per-prompt metrics
            for i in range(len(prompts)):
                prompt = prompts[i]
                prompt_trackers[prompt].update(
                    logits[i:i+1],
                    masks[i:i+1]
                )
            
            # Save predictions
            if save_predictions:
                for i in range(len(images)):
                    save_prediction_mask(
                        preds[i, 0],
                        self.output_dir,
                        image_ids[i],
                        prompts[i]
                    )
            
            # Store some predictions for visualization
            for i in range(len(images)):
                prompt = prompts[i]
                if len(predictions_for_viz[prompt]) < config.NUM_VISUAL_EXAMPLES:
                    predictions_for_viz[prompt].append({
                        'image': images[i].cpu(),
                        'gt_mask': masks[i].cpu(),
                        'pred_mask': preds[i].cpu(),
                        'prompt': prompt,
                        'image_id': image_ids[i]
                    })
        
        # Compute overall metrics
        overall_metrics = overall_tracker.get_metrics()
        overall_std = overall_tracker.get_std()
        
        # Compute per-prompt metrics
        prompt_metrics = {}
        for prompt, tracker in prompt_trackers.items():
            prompt_metrics[prompt] = tracker.get_metrics()
        
        # Compute average inference time
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        # Prepare results
        results = {
            'overall_metrics': overall_metrics,
            'overall_std': overall_std,
            'prompt_metrics': prompt_metrics,
            'avg_inference_time': avg_inference_time,
            'total_samples': len(self.test_loader.dataset),
            'predictions_for_viz': predictions_for_viz
        }
        
        # Print results
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"\nOverall Metrics (n={results['total_samples']}):")
        print(f"  mIoU: {overall_metrics['iou']:.4f} ± {overall_std['iou_std']:.4f}")
        print(f"  Dice: {overall_metrics['dice']:.4f} ± {overall_std['dice_std']:.4f}")
        print(f"  Precision: {overall_metrics['precision']:.4f} ± {overall_std['precision_std']:.4f}")
        print(f"  Recall: {overall_metrics['recall']:.4f} ± {overall_std['recall_std']:.4f}")
        print(f"  F1: {overall_metrics['f1']:.4f} ± {overall_std['f1_std']:.4f}")
        
        print(f"\nPer-Prompt Metrics:")
        for prompt, metrics in prompt_metrics.items():
            print(f"\n  '{prompt}':")
            print(f"    mIoU: {metrics['iou']:.4f}")
            print(f"    Dice: {metrics['dice']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")
        
        print(f"\nInference Time:")
        print(f"  Average: {avg_inference_time*1000:.2f} ms/image")
        
        if save_predictions:
            print(f"\nPredictions saved to: {self.output_dir}")
        
        print("="*60)
        
        return results
    
    def save_results(self, results, output_file=None):
        """Save evaluation results to JSON"""
        if output_file is None:
            output_file = config.LOGS_DIR / "evaluation_results.json"
        
        # Remove predictions_for_viz (not JSON serializable)
        results_to_save = {
            'overall_metrics': results['overall_metrics'],
            'overall_std': results['overall_std'],
            'prompt_metrics': results['prompt_metrics'],
            'avg_inference_time': results['avg_inference_time'],
            'total_samples': results['total_samples']
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate text-conditioned segmentation model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(config.CHECKPOINTS_DIR / "best_model.pth"),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(config.PREDICTIONS_DIR),
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save prediction masks'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(args.checkpoint, args.output_dir)
    
    # Evaluate
    results = evaluator.evaluate(save_predictions=not args.no_save)
    
    # Save results
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
