# src/config.py
import os
from pathlib import Path

# ============ PATHS ============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# VREED and Cognitive Load Paths
VREED_DATA_PATH = DATA_ROOT / "vreed" / "04 Eye Tracking Data" / "02 Eye Tracking Data (Features Extracted)" / "EyeTracking_FeaturesExtracted.csv"
COGLOAD_DATA_PATH = DATA_ROOT / "cognitive_load" / "cognitive_load_dataset.csv"

for directory in [PROCESSED_DATA_ROOT, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============ EXISTING MPIIGAZE SETTINGS ============
SEQUENCE_LENGTH = 32
NUM_FEATURES = 28
ATTENTION_NUM_CLASSES = 3
GAZE_OUTPUT_DIM = 2

# ADDED BACK: Attention Labels required by main.py
ATTENTION_LABELS = {'focused': 0, 'distracted': 1, 'sleeping': 2}

# ============ NEW MODEL SETTINGS ============
EMOTION_NUM_CLASSES = 4  
COGLOAD_NUM_CLASSES = 3

# Emotion Labels mapped to VREED's 4 Quadrant Categories
EMOTION_LABELS = {
    0: 'High Arousal High Valence (Excited/Happy)', 
    1: 'Low Arousal High Valence (Calm/Relaxed)', 
    2: 'Low Arousal Low Valence (Sad/Bored)', 
    3: 'High Arousal Low Valence (Angry/Anxious)'
}

# Cognitive Load Labels (0-back, 1-back, 2-back)
COGLOAD_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# ============ TRAINING SETTINGS ============
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

print("[OK] Configuration loaded with updated Paths and Classes")