"""
Evaluation script for trained models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import *
from feature_engineering import FeatureExtractor
from sklearn.model_selection import train_test_split

class EvaluationPipeline:
    """Evaluate trained models"""
    
    def __init__(self):
        self.sequences = None
    
    def load_sequences(self):
        """Load preprocessed sequences"""
        sequences_file = PROCESSED_DATA_ROOT / "sequences.pkl"
        
        with open(sequences_file, 'rb') as f:
            self.sequences = pickle.load(f)
        
        return self.sequences
    
    def prepare_test_data(self):
        """Prepare test data (same split as training)"""
        if self.sequences is None:
            self.load_sequences()
        
        # Extract features (same as training)
        X_features = []
        y_attention = []
        y_gaze = []
        
        for seq in self.sequences:
            features = FeatureExtractor.extract_temporal_features(seq)
            X_features.append(features)
            y_attention.append(seq['attention_label'])
            gaze_mean = np.mean(seq['gaze_targets'], axis=0)
            y_gaze.append(gaze_mean)
        
        X_features = np.array(X_features)
        y_attention = np.array(y_attention)
        y_gaze = np.array(y_gaze)
        
        # Split (same random seed as training)
        test_size = TEST_SPLIT / (1 - VAL_SPLIT)
        X_temp, X_test, y_attn_temp, y_attn_test, y_gaze_temp, y_gaze_test = \
            train_test_split(
                X_features, y_attention, y_gaze,
                test_size=test_size,
                random_state=42
            )
        
        return X_test, y_attn_test, y_gaze_test
    
    def evaluate(self):
        """Evaluate both models"""
        print("=" * 70)
        print("EVALUATION")
        print("=" * 70)
        
        # Load test data
        X_test, y_attn_test, y_gaze_test = self.prepare_test_data()
        
        # Load models
        print("\nLoading trained models...")
        model_attn = keras.models.load_model(MODELS_DIR / 'attention_classifier.h5')
        model_gaze = keras.models.load_model(MODELS_DIR / 'gaze_estimator.h5')
        
        results = {}
        
        # Evaluate attention classifier
        print("\n----- ATTENTION CLASSIFIER EVALUATION -----")
        loss_attn, acc_attn = model_attn.evaluate(X_test, y_attn_test, verbose=0)
        
        y_pred_attn = model_attn.predict(X_test)
        y_pred_attn_classes = np.argmax(y_pred_attn, axis=1)
        
        print(f"Test Accuracy: {acc_attn*100:.2f}%")
        print(f"Test Loss: {loss_attn:.4f}")
        
        # Classification report
        report_attn = classification_report(
            y_attn_test, y_pred_attn_classes,
            target_names=['Focused', 'Distracted', 'Sleeping'],
            output_dict=True
        )
        print(classification_report(
            y_attn_test, y_pred_attn_classes,
            target_names=['Focused', 'Distracted', 'Sleeping']
        ))
        
        results['attention'] = {
            'accuracy': float(acc_attn),
            'loss': float(loss_attn),
            'report': report_attn
        }
        
        # Confusion matrix
        cm_attn = confusion_matrix(y_attn_test, y_pred_attn_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_attn, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Focused', 'Distracted', 'Sleeping'],
                    yticklabels=['Focused', 'Distracted', 'Sleeping'])
        plt.title('Attention Classifier Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'confusion_matrix_attention.png')
        plt.close()
        
        # Evaluate gaze estimator
        print("\n----- GAZE ESTIMATOR EVALUATION -----")
        loss_gaze, mae_gaze = model_gaze.evaluate(X_test, y_gaze_test, verbose=0)
        
        y_pred_gaze = model_gaze.predict(X_test)
        
        # Calculate angular error
        angular_error = np.mean(np.arccos(
            np.clip(np.sum(y_pred_gaze * y_gaze_test, axis=1), -1, 1)
        )) * 180 / np.pi
        
        print(f"Test MSE: {loss_gaze:.6f}")
        print(f"Test MAE: {mae_gaze:.6f}")
        print(f"Angular Error: {angular_error:.2f}°")
        
        results['gaze'] = {
            'mse': float(loss_gaze),
            'mae': float(mae_gaze),
            'angular_error': float(angular_error)
        }
        
        # Save results
        results_file = RESULTS_DIR / 'metrics.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 70)
        print("✓ Evaluation complete")
        print(f"Results saved to {results_file}")
        print("=" * 70)


if __name__ == "__main__":
    evaluator = EvaluationPipeline()
    evaluator.evaluate()
