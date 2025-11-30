"""
Inference script for single images or batches
"""
import os
import sys
from pathlib import Path
import argparse

import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from models.clip_segmentation import build_model
from utils.visualization import save_prediction_mask, visualize_prediction


class Predictor:
    """Predictor class for inference"""
    
    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)
        
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
        
        print("Model loaded successfully!")
        
        # Setup transform
        self.transform = A.Compose([
            A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @torch.no_grad()
    def predict(self, image_path, prompt, threshold=None):
        """
        Predict segmentation mask for a single image
        
        Args:
            image_path: Path to input image
            prompt: Text prompt
            threshold: Threshold for binarizing prediction (default from config)
            
        Returns:
            Dictionary with prediction results
        """
        if threshold is None:
            threshold = config.PREDICTION_THRESHOLD
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]
        
        # Transform
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        logits = self.model(image_tensor, [prompt])
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()
        
        # Resize back to original size
        pred_mask_resized = torch.nn.functional.interpolate(
            pred_mask,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        pred_mask_resized = (pred_mask_resized > 0.5).float()
        
        # Convert to numpy
        pred_mask_np = pred_mask_resized[0, 0].cpu().numpy()
        
        return {
            'image': image_rgb,
            'mask': pred_mask_np,
            'prompt': prompt,
            'image_path': image_path
        }
    
    def predict_batch(self, image_paths, prompts, output_dir=None, visualize=False):
        """
        Predict for multiple images
        
        Args:
            image_paths: List of image paths
            prompts: List of prompts (one per image)
            output_dir: Directory to save predictions
            visualize: Whether to create visualizations
        """
        if len(image_paths) != len(prompts):
            raise ValueError("Number of images and prompts must match")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for image_path, prompt in zip(image_paths, prompts):
            print(f"\nProcessing: {image_path}")
            print(f"Prompt: {prompt}")
            
            # Predict
            result = self.predict(image_path, prompt)
            results.append(result)
            
            # Save mask
            if output_dir:
                image_id = Path(image_path).stem
                save_prediction_mask(
                    result['mask'],
                    output_dir,
                    image_id,
                    prompt
                )
                print(f"Saved prediction to: {output_dir}")
            
            # Visualize
            if visualize and output_dir:
                viz_path = output_dir / f"{image_id}__visualization.png"
                visualize_prediction(
                    result['image'],
                    result['mask'],  # Use as GT for visualization
                    result['mask'],
                    prompt,
                    save_path=viz_path
                )
                print(f"Saved visualization to: {viz_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Inference for text-conditioned segmentation')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(config.CHECKPOINTS_DIR / "best_model.pth"),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Text prompt (e.g., "segment crack" or "segment taping area")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(config.PREDICTIONS_DIR / "inference"),
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=config.PREDICTION_THRESHOLD,
        help='Threshold for binarizing predictions'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization'
    )
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = Predictor(args.checkpoint)
    
    # Predict
    result = predictor.predict(args.image, args.prompt, args.threshold)
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_id = Path(args.image).stem
    save_prediction_mask(
        result['mask'],
        output_dir,
        image_id,
        args.prompt
    )
    
    print(f"\nPrediction saved to: {output_dir}")
    
    # Visualize
    if args.visualize:
        viz_path = output_dir / f"{image_id}__visualization.png"
        visualize_prediction(
            result['image'],
            result['mask'],
            result['mask'],
            args.prompt,
            save_path=viz_path
        )
        print(f"Visualization saved to: {viz_path}")


if __name__ == "__main__":
    main()
