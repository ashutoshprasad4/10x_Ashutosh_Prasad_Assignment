"""
Text-conditioned segmentation model using CLIP + U-Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
import segmentation_models_pytorch as smp


class CLIPSegmentationModel(nn.Module):
    """
    Text-conditioned segmentation model
    
    Architecture:
    1. CLIP text encoder for prompt embeddings
    2. Image encoder (ResNet/EfficientNet backbone)
    3. Feature fusion module
    4. U-Net decoder for segmentation
    """
    
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        freeze_clip=True
    ):
        super(CLIPSegmentationModel, self).__init__()
        
        # CLIP text encoder
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP if specified
        if freeze_clip:
            for param in self.clip_text_encoder.parameters():
                param.requires_grad = False
        
        # Get CLIP embedding dimension
        self.text_embed_dim = self.clip_text_encoder.config.hidden_size
        
        # Image encoder (U-Net with pretrained encoder)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  # We'll apply sigmoid separately
        )
        
        # Get encoder output channels
        if encoder_name.startswith('resnet'):
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name.startswith('efficientnet'):
            encoder_channels = [16, 24, 40, 112, 320]
        else:
            # Default channels
            encoder_channels = [64, 128, 256, 512, 1024]
        
        # Feature fusion modules - inject text features at multiple scales
        self.fusion_modules = nn.ModuleList([
            FeatureFusion(ch, self.text_embed_dim)
            for ch in encoder_channels
        ])
        
        # Cache for text embeddings (to avoid recomputing for same prompts)
        self.text_cache = {}
    
    def encode_text(self, prompts):
        """
        Encode text prompts using CLIP
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Text embeddings (batch_size, text_embed_dim)
        """
        # Tokenize
        inputs = self.clip_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(next(self.clip_text_encoder.parameters()).device)
        
        # Encode
        with torch.set_grad_enabled(self.clip_text_encoder.training):
            outputs = self.clip_text_encoder(**inputs)
            # Use pooled output (CLS token)
            text_embeds = outputs.pooler_output
        
        return text_embeds
    
    def forward(self, images, prompts):
        """
        Forward pass
        
        Args:
            images: Input images (batch_size, 3, H, W)
            prompts: List of text prompts (batch_size,)
            
        Returns:
            Segmentation logits (batch_size, 1, H, W)
        """
        # Encode text prompts
        text_embeds = self.encode_text(prompts)
        
        # Get image features from encoder
        features = self.unet.encoder(images)
        
        # Skip the first feature (input image) as it has 3 channels and we expect 64
        # ResNet encoder returns [input, layer1, layer2, layer3, layer4, layer5]
        features_to_fuse = features[1:]
        
        # Fuse text features with image features at each scale
        fused_features = []
        for i, (feat, fusion_module) in enumerate(zip(features_to_fuse, self.fusion_modules)):
            fused_feat = fusion_module(feat, text_embeds)
            fused_features.append(fused_feat)
            
        # Add the skipped feature back to the list for decoder
        # The decoder expects the full list of features
        all_features = [features[0]] + fused_features
        
        # Decode
        decoder_output = self.unet.decoder(*all_features)
        
        # Segmentation head
        logits = self.unet.segmentation_head(decoder_output)
        
        return logits


class FeatureFusion(nn.Module):
    """
    Fuse image features with text embeddings
    
    Uses FiLM (Feature-wise Linear Modulation) conditioning
    """
    
    def __init__(self, image_channels, text_embed_dim):
        super(FeatureFusion, self).__init__()
        
        # Project text embeddings to generate scale and shift parameters
        self.scale_proj = nn.Linear(text_embed_dim, image_channels)
        self.shift_proj = nn.Linear(text_embed_dim, image_channels)
        
        # Initialize to identity transformation
        self.scale_proj.weight.data.zero_()
        self.scale_proj.bias.data.fill_(1.0)
        self.shift_proj.weight.data.zero_()
        self.shift_proj.bias.data.zero_()
    
    def forward(self, image_features, text_embeds):
        """
        Apply FiLM conditioning
        
        Args:
            image_features: (batch_size, channels, H, W)
            text_embeds: (batch_size, text_embed_dim)
            
        Returns:
            Conditioned features (batch_size, channels, H, W)
        """
        # Generate scale and shift
        scale = self.scale_proj(text_embeds)  # (batch_size, channels)
        shift = self.shift_proj(text_embeds)  # (batch_size, channels)
        
        # Reshape for broadcasting
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        
        # Apply FiLM: y = scale * x + shift
        conditioned = scale * image_features + shift
        
        return conditioned


def build_model(config_dict=None):
    """
    Build the segmentation model
    
    Args:
        config_dict: Optional configuration dictionary
        
    Returns:
        Model instance
    """
    import config
    
    if config_dict is None:
        config_dict = {
            'clip_model_name': config.CLIP_MODEL_NAME,
            'encoder_name': config.ENCODER_NAME,
            'encoder_weights': config.ENCODER_WEIGHTS,
            'in_channels': config.INPUT_CHANNELS,
            'classes': config.OUTPUT_CHANNELS,
        }
    
    model = CLIPSegmentationModel(**config_dict)
    
    return model


if __name__ == "__main__":
    # Test model
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    import config
    
    print("Testing model...")
    
    # Create model
    model = build_model()
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512)
    prompts = ["segment crack", "segment taping area"]
    
    # Forward pass
    with torch.no_grad():
        logits = model(images, prompts)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Prompts: {prompts}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Model test passed!")
