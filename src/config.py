"""
Configuration file for MPIIGaze project
All settings go here - change these for different experiments
"""

import os
from pathlib import Path

# ============ PATHS ============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_ROOT, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============ DATASET ============
NUM_PARTICIPANTS = 15  # Change to 2 for testing (faster)
PARTICIPANTS = [f"p{i:02d}" for i in range(NUM_PARTICIPANTS)]

# ============ IMAGE SETTINGS ============
IMG_HEIGHT = 36
IMG_WIDTH = 60
IMG_CHANNELS = 3

# ============ MODEL SETTINGS ============
SEQUENCE_LENGTH = 32  # How many frames per sequence (1 second at 30fps)
NUM_FEATURES = 28    # Features extracted per frame

# Output sizes
ATTENTION_NUM_CLASSES = 3  # Focused, Distracted, Sleeping
GAZE_OUTPUT_DIM = 2        # x, y coordinates

# ============ TRAINING SETTINGS ============
BATCH_SIZE = 32        # How many samples per training batch
EPOCHS = 100           # How many times to go through data
LEARNING_RATE = 0.001  # How fast model learns
VAL_SPLIT = 0.15       # 15% for validation
TEST_SPLIT = 0.15      # 15% for testing

# ============ EARLY STOPPING ============
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
REDUCE_LR_PATIENCE = 5         # Reduce learning rate after 5 epochs
REDUCE_LR_FACTOR = 0.5         # Multiply learning rate by 0.5

# ============ DATA AUGMENTATION ============
AUGMENTATION_ENABLED = True
TEMPORAL_JITTER_RANGE = 0.1    # ±10% time shift
FEATURE_NOISE_STD = 0.05       # ±5% Gaussian noise

# ============ SCREEN CALIBRATION ============
SCREEN_HEIGHT_PIXEL = 1080
SCREEN_WIDTH_PIXEL = 1920
SCREEN_HEIGHT_MM = 336
SCREEN_WIDTH_MM = 597

# ============ GAZE NORMALIZATION ============
GAZE_NORM_MIN = 0
GAZE_NORM_MAX = 1

# ============ ATTENTION LABELS ============
ATTENTION_LABELS = {
    'focused': 0,      # Normal attention
    'distracted': 1,   # Looking away, fidgeting
    'sleeping': 2      # Eyes closed, very low movement
}

ATTENTION_LABELS_REVERSE = {v: k for k, v in ATTENTION_LABELS.items()}

print(f"✓ Configuration loaded")
print(f"  Data root: {DATA_ROOT}")
print(f"  Processed data: {PROCESSED_DATA_ROOT}")
print(f"  Participants: {NUM_PARTICIPANTS}")
