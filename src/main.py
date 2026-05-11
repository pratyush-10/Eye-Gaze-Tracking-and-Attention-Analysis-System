# src/main.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *

# Reverse dictionaries to get string labels from prediction indices
REVERSE_ATTENTION = {v: k for k, v in ATTENTION_LABELS.items()}
REVERSE_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}
REVERSE_COGLOAD = {0: "Low (0-back)", 1: "Medium (1-back)", 2: "High (2-back)"}

def load_all_models():
    print("Loading all 4 models... This may take a moment.")
    models = {}
    try:
        models['gaze'] = keras.models.load_model(MODELS_DIR / 'gaze_estimator.keras')
        models['attention'] = keras.models.load_model(MODELS_DIR / 'attention_classifier.keras')
        models['emotion'] = keras.models.load_model(MODELS_DIR / 'emotion_classifier.h5')
        models['cogload'] = keras.models.load_model(MODELS_DIR / 'cognitive_load_classifier.h5')
        print("✓ All 4 models loaded successfully!")
        return models
    except Exception as e:
        print(f"Error loading models: {e}\nPlease make sure all 4 models have been trained and are in the 'models/' folder.")
        return None

def run_multimodal_inference():
    models = load_all_models()
    if not models:
        return

    print("\n" + "="*50)
    print("   MULTIMODAL SYSTEM SIMULATION (ALL 4 MODELS)   ")
    print("="*50)

    # ---------------------------------------------------------
    # SIMULATE INCOMING DATA STREAMS 
    # (In a real system, this comes from live sensors/cameras)
    # ---------------------------------------------------------
    
    # 1. Image Sequence features for Gaze & Attention (Shape: 1 batch, 32 frames, 21 features)
    # Using 28 features as dictated by your original config NUM_FEATURES
    dummy_video_sequence = np.random.rand(1, SEQUENCE_LENGTH, NUM_FEATURES) 
    
    # 2. Eye Tracking CSV features for Emotion
    # (Assuming the VREED feature extractor outputs roughly ~30-40 features)
    dummy_emotion_features = np.random.rand(1, models['emotion'].input_shape[1]) 
    
    # 3. Multimodal EEG/fNIRS CSV features for Cognitive Load (414 features expected)
    dummy_cogload_features = np.random.rand(1, models['cogload'].input_shape[1])

    # ---------------------------------------------------------
    # SIMULTANEOUS PREDICTIONS
    # ---------------------------------------------------------
    # Gaze & Attention
    gaze_pred = models['gaze'].predict(dummy_video_sequence, verbose=0)[0]
    attn_pred = models['attention'].predict(dummy_video_sequence, verbose=0)
    attn_class = REVERSE_ATTENTION[np.argmax(attn_pred)]

    # Emotion
    emo_pred = models['emotion'].predict(dummy_emotion_features, verbose=0)
    emo_class = REVERSE_EMOTION[np.argmax(emo_pred)]

    # Cognitive Load
    cog_pred = models['cogload'].predict(dummy_cogload_features, verbose=0)
    cog_class = REVERSE_COGLOAD[np.argmax(cog_pred)]

    # ---------------------------------------------------------
    # OUTPUT RESULTS
    # ---------------------------------------------------------
    print("\n[ REAL-TIME ANALYSIS RESULTS ]")
    print(f"➤ Gaze Position    : [X: {gaze_pred[0]:.3f}, Y: {gaze_pred[1]:.3f}]")
    print(f"➤ Attention Level  : {attn_class.upper()}")
    print(f"➤ Emotion State    : {emo_class.upper()}")
    print(f"➤ Cognitive Load   : {cog_class.upper()}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_multimodal_inference()