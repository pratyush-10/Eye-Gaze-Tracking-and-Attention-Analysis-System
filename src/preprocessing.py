"""
Preprocessing pipeline
Converts raw images and annotations into training data
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from sklearn.model_selection import train_test_split
from config import *
from data_loader import MPIIGazeDataLoader
import scipy.io


class PreprocessingPipeline:
    """Preprocess MPIIGaze data for model training"""
    
    def __init__(self, output_dir=PROCESSED_DATA_ROOT):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = MPIIGazeDataLoader()
        self.calibration_path = Path(DATA_ROOT) / "Calibration"
    
    def normalize_gaze_coordinates(self, gaze_coords, participant_id):
        """Silent MPIIGaze calibration parser."""
        
        # Handle 1D → 2D conversion
        original_shape = gaze_coords.shape
        if gaze_coords.ndim == 1:
            gaze_coords = gaze_coords.reshape(-1, 2)
        
        # Try calibration files
        calibration_file = self.calibration_path / participant_id / "Camera.mat"
        if not calibration_file.exists():
            calibration_file = self.calibration_path / participant_id / "screenSize.mat"
        
        if calibration_file.exists():
            try:
                data = scipy.io.loadmat(str(calibration_file))
                
                # Try common MPIIGaze locations
                if 'cam' in data:
                    cam = data['cam'][0][0]
                    if 'screen_width_pixel' in cam.dtype.fields:
                        w = float(cam['screen_width_pixel'][0][0][0][0])
                        h = float(cam['screen_height_pixel'][0][0][0][0])
                    else:
                        return self._default_normalize(gaze_coords)
                elif 'screen_width_pixel' in data:
                    w = float(data['screen_width_pixel'][0][0])
                    h = float(data['screen_height_pixel'][0][0])
                else:
                    return self._default_normalize(gaze_coords)
                
                # Success: normalize with real values
                normalized = np.zeros_like(gaze_coords)
                normalized[:, 0] = np.clip(gaze_coords[:, 0] / w, 0, 1)
                normalized[:, 1] = np.clip(gaze_coords[:, 1] / h, 0, 1)
                return normalized
                
            except:
                pass  # Silent fail
        
        # Default fallback (no print)
        return self._default_normalize(gaze_coords)

    def _default_normalize(self, gaze_coords):
        """Safe default 1920x1080."""
        if gaze_coords.ndim == 1:
            gaze_coords = gaze_coords.reshape(-1, 2)
        
        w, h = 1920.0, 1080.0
        normalized = np.zeros_like(gaze_coords)
        normalized[:, 0] = np.clip(gaze_coords[:, 0] / w, 0, 1)
        normalized[:, 1] = np.clip(gaze_coords[:, 1] / h, 0, 1)
        return normalized
    
    def create_attention_labels(self, samples):
        """
        Create attention labels based on gaze patterns
        - Low movement = Sleeping (2)
        - Medium movement = Distracted (1)
        - High movement = Focused (0)
        """
        attention_labels = []
        gaze_velocities = []
        
        # Calculate gaze velocity
        for idx, sample in enumerate(samples):
            if idx == 0:
                gaze_velocities.append(0)
            else:
                prev_gaze = samples[idx-1]['gaze_2d_normalized']
                curr_gaze = sample['gaze_2d_normalized']
                velocity = np.linalg.norm(curr_gaze - prev_gaze)
                gaze_velocities.append(velocity)
        
        gaze_velocities = np.array(gaze_velocities)
        
        # Classify based on movement
        for idx, (sample, velocity) in enumerate(zip(samples, gaze_velocities)):
            head_pose = sample['annotation']['head_pose']
            head_movement = np.linalg.norm(head_pose[:3])
            
            combined_motion = velocity + 0.1 * head_movement
            
            if combined_motion < 0.05:
                label = 2  # Sleeping
            elif combined_motion < 0.15:
                label = 1  # Distracted
            else:
                label = 0  # Focused
            
            attention_labels.append(label)
        
        return np.array(attention_labels)
    
    def process_all_data(self):
        """Main preprocessing pipeline"""
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Collect all samples
        print("\n[1/5] Collecting all samples...")
        samples = self.loader.collect_all_samples()
        
        # Step 2: Load images and extract features
        print(f"\n[2/5] Loading images and extracting features...")
        processed_samples = []
        
        for sample in tqdm(samples, desc="Processing samples"):
            # Load image
            image = self.loader.load_image(sample['image_path'])
            
            # Extract annotation
            ann = sample['annotation']
            calib = sample['calibration']
            
            # Normalize gaze
            gaze_2d_normalized = self.normalize_gaze_coordinates(
                ann['gaze_2d_screen'], sample['participant_id']
            )
            
            # Create processed sample
            processed = {
                'participant_id': sample['participant_id'],
                'day': sample['day'],
                'image_index': sample['image_index'],
                'image': image,
                'gaze_2d_normalized': gaze_2d_normalized,
                'head_pose': ann['head_pose'],
                'eye_landmarks': ann['eye_landmarks'],
                'annotation': ann,
                'calibration': calib
            }
            
            processed_samples.append(processed)
        
        # Step 3: Create attention labels
        print(f"\n[3/5] Creating attention labels...")
        attention_labels = self.create_attention_labels(processed_samples)
        
        for sample, label in zip(processed_samples, attention_labels):
            sample['attention_label'] = label
        
        # Step 4: Create temporal sequences
        print(f"\n[4/5] Creating temporal sequences...")
        sequences = self.create_temporal_sequences(processed_samples)
        
        print(f"  Created {len(sequences)} sequences from {len(processed_samples)} samples")
        
        # Step 5: Save processed data
        print(f"\n[5/5] Saving processed data...")
        self._save_processed_data(processed_samples, sequences)
        
        print("\n" + "=" * 60)
        print("✓ Preprocessing complete")
        print("=" * 60)
        
        return processed_samples, sequences
    
    def create_temporal_sequences(self, samples):
        """Group consecutive frames into sequences"""
        sequences = []
        
        # Group samples by participant and day
        participant_day_groups = {}
        
        for sample in samples:
            key = (sample['participant_id'], sample['day'])
            if key not in participant_day_groups:
                participant_day_groups[key] = []
            participant_day_groups[key].append(sample)
        
        # Create sequences with sliding window
        for (participant, day), day_samples in participant_day_groups.items():
            # Sort by image index
            day_samples = sorted(day_samples, key=lambda x: x['image_index'])
            
            # Create sequences with 50% overlap
            stride = SEQUENCE_LENGTH // 2
            
            for start_idx in range(0, len(day_samples) - SEQUENCE_LENGTH, stride):
                seq_samples = day_samples[start_idx:start_idx + SEQUENCE_LENGTH]
                
                # Gather data
                images = np.array([s['image'] for s in seq_samples])
                gaze_targets = np.array([s['gaze_2d_normalized'] for s in seq_samples])
                head_poses = np.array([s['head_pose'] for s in seq_samples])
                
                # Use most common attention label in sequence
                attention_labels_in_seq = [s['attention_label'] for s in seq_samples]
                attention_label = np.bincount(attention_labels_in_seq).argmax()
                
                sequence = {
                    'participant_id': participant,
                    'day': day,
                    'image_indices': list(range(start_idx, start_idx + SEQUENCE_LENGTH)),
                    'images': images,
                    'gaze_targets': gaze_targets,
                    'head_poses': head_poses,
                    'attention_label': attention_label
                }
                
                sequences.append(sequence)
        
        return sequences
    
    def _save_processed_data(self, processed_samples, sequences):
        """Save processed data to disk"""
        # Save sequences
        sequences_file = self.output_dir / "sequences.pkl"
        with open(sequences_file, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"  Saved {len(sequences)} sequences to {sequences_file}")
        
        # Save metadata
        metadata = []
        for seq in sequences:
            meta = {
                'participant_id': seq['participant_id'],
                'day': seq['day'],
                'image_indices': seq['image_indices'],
                'attention_label': ATTENTION_LABELS_REVERSE[seq['attention_label']]
            }
            metadata.append(meta)
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata to {metadata_file}")

# Run when called directly
if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    processed_samples, sequences = pipeline.process_all_data()
