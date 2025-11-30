"""
Configuration file for Prompted Segmentation for Drywall QA
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset paths
DATASET1_DIR = DATA_DIR / "drywall_join_detect"  # Taping areas
DATASET2_DIR = DATA_DIR / "cracks"  # Cracks

# ============================================================================
# ROBOFLOW CONFIGURATION
# ============================================================================
# Dataset 1: Drywall-Join-Detect (Taping areas)
DATASET1_WORKSPACE = "objectdetect-pu6rn"
DATASET1_PROJECT = "drywall-join-detect"
DATASET1_VERSION = 1

# Dataset 2: Cracks
DATASET2_WORKSPACE = "fyp-ny1jt"
DATASET2_PROJECT = "cracks-3ii36"
DATASET2_VERSION = 1

# Roboflow API key (set via environment variable or .env file)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

# ============================================================================
# PROMPT MAPPINGS
# ============================================================================
# Map dataset to possible prompts
PROMPT_MAPPINGS = {
    "taping": [
        "segment taping area",
        "segment joint",
        "segment tape",
        "segment drywall seam",
    ],
    "crack": [
        "segment crack",
        "segment wall crack",
        "segment cracks",
    ]
}

# Primary prompts for training and evaluation
PRIMARY_PROMPTS = {
    "taping": "segment taping area",
    "crack": "segment crack"
}

# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Image settings
IMAGE_SIZE = (512, 512)  # (height, width)
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1  # Binary segmentation

# CLIP model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512

# Encoder backbone
ENCODER_NAME = "resnet34"  # Options: resnet34, resnet50, efficientnet-b0, etc.
ENCODER_WEIGHTS = "imagenet"

# Decoder
DECODER_CHANNELS = [256, 128, 64, 32, 16]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Optimizer
OPTIMIZER = "AdamW"

# Learning rate scheduler
LR_SCHEDULER = "ReduceLROnPlateau"
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-7

# Loss weights
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# Mixed precision training
USE_AMP = True

# Gradient clipping
GRAD_CLIP_MAX_NORM = 1.0

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
AUGMENTATION_CONFIG = {
    "train": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.3,
        "rotate_limit": 15,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "gaussian_blur": 0.3,
        "gaussian_noise": 0.2,
    },
    "val": {},  # No augmentation for validation
    "test": {}  # No augmentation for test
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Metrics to compute
METRICS = ["iou", "dice", "precision", "recall", "f1"]

# Threshold for binary prediction
PREDICTION_THRESHOLD = 0.5

# Number of visual examples to generate
NUM_VISUAL_EXAMPLES = 4

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
# Output format for prediction masks
MASK_OUTPUT_FORMAT = "png"
MASK_VALUES = [0, 255]  # Background: 0, Foreground: 255

# Filename format: {image_id}__{prompt}.png
# Example: 123__segment_crack.png

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # For DataLoader

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_directories():
    """Create necessary directories if they don't exist"""
    for directory in [DATA_DIR, CHECKPOINTS_DIR, PREDICTIONS_DIR, LOGS_DIR, TENSORBOARD_LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_prompt_for_dataset(dataset_type):
    """Get primary prompt for a dataset type"""
    return PRIMARY_PROMPTS.get(dataset_type, "")

def get_all_prompts_for_dataset(dataset_type):
    """Get all possible prompts for a dataset type"""
    return PROMPT_MAPPINGS.get(dataset_type, [])
