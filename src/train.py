"""
Training script for attention and gaze models
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


from config import *
from models import build_attention_classifier, build_gaze_estimator
from feature_engineering import FeatureExtractor


class CustomModelCheckpoint(keras.callbacks.Callback):
    """Custom checkpoint callback to avoid Keras format issues"""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='auto'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        
        if mode == 'auto':
            if 'acc' in monitor or 'accuracy' in monitor:
                self.mode = 'max'
            else:
                self.mode = 'min'
        else:
            self.mode = mode
        
        if self.mode == 'min':
            self.best = np.Inf
            self.monitor_op = np.less
        else:
            self.best = -np.Inf
            self.monitor_op = np.greater
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                self.model.save(self.filepath, save_format='keras')
        else:
            self.model.save(self.filepath, save_format='keras')


class TrainingPipeline:
    """Manage model training"""
    
    def __init__(self):
        self.sequences = None
        self.results = {}
    
    def load_sequences(self):
        """Load preprocessed sequences"""
        sequences_file = PROCESSED_DATA_ROOT / "sequences.pkl"
        
        print(f"Loading sequences from {sequences_file}...")
        with open(sequences_file, 'rb') as f:
            self.sequences = pickle.load(f)
        
        print(f"✓ Loaded {len(self.sequences)} sequences")
        return self.sequences
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\n" + "=" * 70)
        print("PREPARING TRAINING DATA")
        print("=" * 70)
        
        if self.sequences is None:
            self.load_sequences()
        
        print(f"\nExtracting temporal features...")
        
        X_features = []
        y_attention = []
        y_gaze = []
        
        for seq in tqdm(self.sequences, desc="Processing sequences"):
            # Extract temporal features
            features = FeatureExtractor.extract_temporal_features(seq)
            
            # Apply augmentation
            if np.random.random() < 0.3 and AUGMENTATION_ENABLED:
                features = FeatureExtractor.augment_temporal_features(features)
            
            X_features.append(features)
            y_attention.append(seq['attention_label'])
            
            # For gaze, use mean over sequence
            gaze_mean = np.mean(seq['gaze_targets'], axis=0)
            y_gaze.append(gaze_mean)
        
        X_features = np.array(X_features, dtype=np.float32)
        y_attention = np.array(y_attention, dtype=np.int32)
        y_gaze = np.array(y_gaze, dtype=np.float32)
        
        print(f"✓ Extracted features: {X_features.shape}")
        
        # Split into train/val/test
        print(f"\nSplitting data (70/15/15)...")
        
        # First split: separate test set
        test_size = TEST_SPLIT / (1 - VAL_SPLIT)
        X_temp, X_test, y_attn_temp, y_attn_test, y_gaze_temp, y_gaze_test = \
            train_test_split(
                X_features, y_attention, y_gaze,
                test_size=test_size,
                random_state=42
            )
        
        # Second split: separate val from train
        val_size = VAL_SPLIT / (VAL_SPLIT + (1 - VAL_SPLIT - TEST_SPLIT))
        X_train, X_val, y_attn_train, y_attn_val, y_gaze_train, y_gaze_val = \
            train_test_split(
                X_temp, y_attn_temp, y_gaze_temp,
                test_size=val_size,
                random_state=42
            )
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        return (X_train, y_attn_train, y_gaze_train,
                X_val, y_attn_val, y_gaze_val,
                X_test, y_attn_test, y_gaze_test)
    
    def train_models(self):
        """Train both models"""
        # Prepare data
        X_train, y_attn_train, y_gaze_train, \
        X_val, y_attn_val, y_gaze_val, \
        X_test, y_attn_test, y_gaze_test = self.prepare_data()
        
        print("\n" + "=" * 70)
        print("TRAINING MODELS")
        print("=" * 70)
        
        # Train attention classifier
        print("\n----- TRAINING ATTENTION CLASSIFIER -----")
        model_attn = build_attention_classifier()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        # Use custom checkpoint callback
        model_checkpoint = CustomModelCheckpoint(
            filepath=str(MODELS_DIR / 'attention_classifier.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        # Train
        history_attn = model_attn.fit(
            X_train, y_attn_train,
            validation_data=(X_val, y_attn_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, model_checkpoint],
            verbose=1
        )
        
        print("✓ Attention classifier trained")
        
        # Save final model using the new Keras format
        model_attn.save(str(MODELS_DIR / 'attention_model.keras'), save_format='keras')
        print(f"✓ Saved: {MODELS_DIR / 'attention_model.keras'}")
        
        # Train gaze estimator
        print("\n----- TRAINING GAZE ESTIMATOR -----")
        model_gaze = build_gaze_estimator()
        
        # Use custom checkpoint callback for gaze model
        model_checkpoint_gaze = CustomModelCheckpoint(
            filepath=str(MODELS_DIR / 'gaze_estimator.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        history_gaze = model_gaze.fit(
            X_train, y_gaze_train,
            validation_data=(X_val, y_gaze_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, model_checkpoint_gaze],
            verbose=1
        )
        
        print("✓ Gaze estimator trained")
        
        # Save final model
        model_gaze.save(str(MODELS_DIR / 'gaze_model.keras'), save_format='keras')
        print(f"✓ Saved: {MODELS_DIR / 'gaze_model.keras'}")
        
        # Save training histories
        self._save_histories(history_attn, history_gaze)
        
        print("\n" + "=" * 70)
        print("✓ Training complete")
        print("=" * 70)
        
        return model_attn, model_gaze
    
    def _save_histories(self, history_attn, history_gaze):
        """Save training histories to JSON"""
        histories = {
            'attention': {
                'loss': [float(x) for x in history_attn.history['loss']],
                'accuracy': [float(x) for x in history_attn.history['accuracy']],
                'val_loss': [float(x) for x in history_attn.history['val_loss']],
                'val_accuracy': [float(x) for x in history_attn.history['val_accuracy']]
            },
            'gaze': {
                'loss': [float(x) for x in history_gaze.history['loss']],
                'val_loss': [float(x) for x in history_gaze.history['val_loss']]
            }
        }
        
        history_file = MODELS_DIR / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(histories, f, indent=2)
        
        print(f"✓ Saved training history: {history_file}")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    model_attn, model_gaze = pipeline.train_models()