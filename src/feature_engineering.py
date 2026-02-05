"""
Feature engineering
Extract temporal features from raw image sequences
"""


import numpy as np
import cv2
from scipy import stats
from config import *


class FeatureExtractor:
    """Extract features from eye images and annotations"""
    
    @staticmethod
    def extract_temporal_features(sequence_data):
        """
        Extract temporal features from a sequence of frames
        
        Input: sequence_data dict with gaze_targets, head_poses
        Output: (SEQUENCE_LENGTH, NUM_FEATURES) array of shape (32, 21)
        """
        try:
            # Convert to numpy arrays immediately
            gaze_targets = np.array(sequence_data['gaze_targets'], dtype=np.float32)
            head_poses = np.array(sequence_data['head_poses'], dtype=np.float32)
            
            # Validate inputs
            if gaze_targets.size == 0 or head_poses.size == 0:
                return np.zeros((SEQUENCE_LENGTH, 21), dtype=np.float32)
            
            # Ensure gaze_targets is 2D
            if gaze_targets.ndim == 1:
                gaze_targets = np.column_stack([gaze_targets, gaze_targets])
            
            if gaze_targets.ndim > 2:
                gaze_targets = gaze_targets.reshape(gaze_targets.shape[0], -1)
            
            if gaze_targets.shape[1] == 1:
                gaze_targets = np.column_stack([gaze_targets, gaze_targets])
            elif gaze_targets.shape[1] > 2:
                gaze_targets = gaze_targets[:, :2]
            
            # Ensure head_poses is 2D
            if head_poses.ndim == 1:
                head_poses = head_poses.reshape(-1, 1)
            
            if head_poses.ndim > 2:
                head_poses = head_poses.reshape(head_poses.shape[0], -1)
            
            # Pad head_poses to 6 features if needed
            if head_poses.shape[1] < 6:
                padding = np.zeros((head_poses.shape[0], 6 - head_poses.shape[1]), dtype=np.float32)
                head_poses = np.column_stack([head_poses, padding])
            elif head_poses.shape[1] > 6:
                head_poses = head_poses[:, :6]
            
            # Sync lengths
            min_len = min(gaze_targets.shape[0], head_poses.shape[0])
            gaze_targets = gaze_targets[:min_len]
            head_poses = head_poses[:min_len]
            
            # Pad/trim to SEQUENCE_LENGTH
            seq_len = min_len
            if seq_len < SEQUENCE_LENGTH:
                # Pad with last frame
                pad_size = SEQUENCE_LENGTH - seq_len
                gaze_targets = np.vstack([
                    gaze_targets,
                    np.repeat(gaze_targets[-1:], pad_size, axis=0)
                ])
                head_poses = np.vstack([
                    head_poses,
                    np.repeat(head_poses[-1:], pad_size, axis=0)
                ])
            elif seq_len > SEQUENCE_LENGTH:
                gaze_targets = gaze_targets[:SEQUENCE_LENGTH]
                head_poses = head_poses[:SEQUENCE_LENGTH]
            
            features_seq = []
            
            for t in range(SEQUENCE_LENGTH):
                features = []
                
                # 1. Gaze position (2 features)
                gaze_x = float(gaze_targets[t, 0])
                gaze_y = float(gaze_targets[t, 1])
                features.append(gaze_x)
                features.append(gaze_y)
                
                # 2. Head pose (6 features)
                for i in range(6):
                    features.append(float(head_poses[t, i]))
                
                # 3. Gaze velocity (2 features)
                if t > 0:
                    gaze_vel_x = float(gaze_targets[t, 0] - gaze_targets[t-1, 0])
                    gaze_vel_y = float(gaze_targets[t, 1] - gaze_targets[t-1, 1])
                    features.append(gaze_vel_x)
                    features.append(gaze_vel_y)
                else:
                    features.append(0.0)
                    features.append(0.0)
                
                # 4. Head pose velocity (6 features)
                if t > 0:
                    for i in range(6):
                        head_vel = float(head_poses[t, i] - head_poses[t-1, i])
                        features.append(head_vel)
                else:
                    for i in range(6):
                        features.append(0.0)
                
                # 5. Gaze acceleration (2 features)
                if t > 1:
                    prev_vel_x = gaze_targets[t-1, 0] - gaze_targets[t-2, 0]
                    prev_vel_y = gaze_targets[t-1, 1] - gaze_targets[t-2, 1]
                    curr_vel_x = gaze_targets[t, 0] - gaze_targets[t-1, 0]
                    curr_vel_y = gaze_targets[t, 1] - gaze_targets[t-1, 1]
                    gaze_accel_x = float(curr_vel_x - prev_vel_x)
                    gaze_accel_y = float(curr_vel_y - prev_vel_y)
                    features.append(gaze_accel_x)
                    features.append(gaze_accel_y)
                else:
                    features.append(0.0)
                    features.append(0.0)
                
                # 6. Fixation indicator (1 feature)
                if t > 0:
                    gaze_diff_x = gaze_targets[t, 0] - gaze_targets[t-1, 0]
                    gaze_diff_y = gaze_targets[t, 1] - gaze_targets[t-1, 1]
                    gaze_speed = float(np.sqrt(gaze_diff_x**2 + gaze_diff_y**2))
                    fixation = 1.0 if gaze_speed < 0.02 else 0.0
                else:
                    fixation = 0.0
                features.append(fixation)
                
                # 7. Head movement magnitude (1 feature)
                head_x = head_poses[t, 0]
                head_y = head_poses[t, 1]
                head_z = head_poses[t, 2]
                head_move = float(np.sqrt(head_x**2 + head_y**2 + head_z**2))
                features.append(head_move)
                
                # 8. Mean gaze velocity over past 5 frames (1 feature)
                if t > 0:
                    past_vels = []
                    for past_t in range(max(0, t-5), t):
                        vel_x = gaze_targets[past_t+1, 0] - gaze_targets[past_t, 0]
                        vel_y = gaze_targets[past_t+1, 1] - gaze_targets[past_t, 1]
                        vel = float(np.sqrt(vel_x**2 + vel_y**2))
                        past_vels.append(vel)
                    mean_past_velocity = float(np.mean(past_vels)) if past_vels else 0.0
                else:
                    mean_past_velocity = 0.0
                features.append(mean_past_velocity)
                
                # Verify exactly 21 features
                if len(features) != 21:
                    print(f"⚠ Warning: Frame {t} has {len(features)} features, expected 21")
                    # Pad with zeros if short
                    while len(features) < 21:
                        features.append(0.0)
                    # Trim if too many
                    features = features[:21]
                
                features_seq.append(features)
            
            # Convert to array
            result = np.array(features_seq, dtype=np.float32)
            
            # Verify shape
            if result.shape != (SEQUENCE_LENGTH, 21):
                print(f"⚠ Shape mismatch: expected {(SEQUENCE_LENGTH, 21)}, got {result.shape}")
                # Force correct shape
                result = result[:SEQUENCE_LENGTH, :21]
            
            return result
        
        except Exception as e:
            print(f"⚠ Error extracting temporal features: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((SEQUENCE_LENGTH, 21), dtype=np.float32)
    
    @staticmethod
    def augment_temporal_features(features_seq):
        """Add random noise for data augmentation"""
        try:
            # Input validation
            if features_seq is None or features_seq.size == 0:
                return features_seq
            
            augmented = features_seq.copy().astype(np.float32)
            
            # Temporal jittering on gaze coordinates only (first 2 features)
            if np.random.random() < 0.5:
                jitter = np.random.uniform(
                    -TEMPORAL_JITTER_RANGE,
                    TEMPORAL_JITTER_RANGE,
                    size=(augmented.shape[0], 2)
                )
                augmented[:, :2] += jitter
                augmented[:, :2] = np.clip(augmented[:, :2], 0, 1)
            
            # Feature noise on all features
            if np.random.random() < 0.5:
                noise = np.random.normal(0, FEATURE_NOISE_STD, size=augmented.shape)
                augmented += noise
            
            return augmented.astype(np.float32)
        
        except Exception as e:
            print(f"⚠ Error in augmentation: {e}")
            return features_seq  # Return original on error
