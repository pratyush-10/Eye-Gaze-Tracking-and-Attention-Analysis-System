import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'models/gaze_model.keras'
DATA_PATH = 'data/processed/sequences.pkl' 

SEQUENCE_LENGTH = 32
NUM_FEATURES = 21
SCREEN_W, SCREEN_H = 1920, 1080 

# ==========================================
# FEATURE EXTRACTOR (Robust Version)
# ==========================================
class OfflineFeatureExtractor:
    @staticmethod
    def extract(sequence):
        # 1. Get raw arrays
        gaze_targets = np.array(sequence['gaze_targets'], dtype=np.float32)
        head_poses = np.array(sequence['head_poses'], dtype=np.float32)
        
        # --- FIX: Squeeze extra dimensions (e.g., (32, 1, 2) -> (32, 2)) ---
        if gaze_targets.ndim > 2:
            gaze_targets = gaze_targets.reshape(len(gaze_targets), -1)
        if head_poses.ndim > 2:
            head_poses = head_poses.reshape(len(head_poses), -1)
            
        # 2. Safety checks (Pad or Trim)
        curr_len = len(gaze_targets)
        if curr_len < SEQUENCE_LENGTH:
            pad_amt = SEQUENCE_LENGTH - curr_len
            gaze_targets = np.pad(gaze_targets, ((0, pad_amt), (0,0)), mode='edge')
            head_poses = np.pad(head_poses, ((0, pad_amt), (0,0)), mode='edge')
        elif curr_len > SEQUENCE_LENGTH:
            gaze_targets = gaze_targets[:SEQUENCE_LENGTH]
            head_poses = head_poses[:SEQUENCE_LENGTH]
            
        features_seq = []

        for t in range(SEQUENCE_LENGTH):
            feats = []
            
            # --- A. Basic Features ---
            # Extend works correctly on 1D arrays (e.g. [x, y])
            feats.extend(gaze_targets[t].flatten()) 
            feats.extend(head_poses[t].flatten())
            
            # --- B. Velocities ---
            if t > 0:
                g_vel = gaze_targets[t] - gaze_targets[t-1]
                h_vel = head_poses[t] - head_poses[t-1]
                feats.extend(g_vel.flatten())
                feats.extend(h_vel.flatten())
            else:
                feats.extend([0.0]*8) # 2+6 zeros

            # --- C. Advanced Features ---
            # Acceleration
            if t > 1:
                v1 = gaze_targets[t] - gaze_targets[t-1]
                v0 = gaze_targets[t-1] - gaze_targets[t-2]
                acc = v1 - v0
                feats.extend(acc.flatten())
            else:
                feats.extend([0.0, 0.0])

            # Fixation (Scalar)
            if t > 0:
                mag = np.linalg.norm(gaze_targets[t] - gaze_targets[t-1])
                feats.append(float(1.0 if mag < 0.02 else 0.0))
            else:
                feats.append(0.0)
                
            # Head Mag (Scalar)
            feats.append(float(np.linalg.norm(head_poses[t][:3])))
            
            # Mean Velocity (Scalar)
            if t > 0:
                start = max(0, t-5)
                vels = np.linalg.norm(gaze_targets[start+1:t+1] - gaze_targets[start:t], axis=1)
                feats.append(float(np.mean(vels)) if len(vels) > 0 else 0.0)
            else:
                feats.append(0.0)
            
            # Pad to 21 features
            if len(feats) < NUM_FEATURES:
                feats.extend([0.0]*(NUM_FEATURES - len(feats)))
            
            # Ensure strictly list of floats
            features_seq.append(feats[:NUM_FEATURES])
            
        return np.array(features_seq, dtype=np.float32)

# ==========================================
# MAIN EVALUATION
# ==========================================
def calculate_accuracy(y_true, y_pred, tolerance=0.10):
    """ % of predictions within tolerance distance """
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    correct = np.sum(distances <= tolerance)
    return (correct / len(distances)) * 100

def main():
    print("="*60)
    print("GAZE EVALUATION (OFFLINE)")
    print("="*60)

    # 1. Load Model
    print(f"[1/4] Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load Data
    print(f"[2/4] Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ File not found: {DATA_PATH}")
        return

    with open(DATA_PATH, 'rb') as f:
        raw_sequences = pickle.load(f)
    
    print(f"      Found {len(raw_sequences)} raw sequences. Processing...")
    
    X_all = []
    y_all = []

    for seq in raw_sequences:
        try:
            # Extract X (Input Features)
            feats = OfflineFeatureExtractor.extract(seq)
            
            # Extract y (Target: Mean Gaze)
            gaze_data = np.array(seq['gaze_targets'])
            if gaze_data.ndim > 2: gaze_data = gaze_data.reshape(len(gaze_data), -1)
            gaze_mean = np.mean(gaze_data, axis=0)
            
            X_all.append(feats)
            y_all.append(gaze_mean)
        except Exception as e:
            # Skip bad sequences
            continue

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    # 3. Split Test Set
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=42)
    print(f"✓ Data processed. Test Set: {len(X_test)} samples.")

    # 4. Inference
    print(f"[3/4] Running inference...")
    y_pred = model.predict(X_test, verbose=1)

    # 5. Metrics
    print(f"[4/4] Calculating metrics...")
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    acc_5 = calculate_accuracy(y_test, y_pred, tolerance=0.05)
    acc_10 = calculate_accuracy(y_test, y_pred, tolerance=0.10)
    
    pixel_err_x = mae * SCREEN_W
    pixel_err_y = mae * SCREEN_H

    print("\n" + "="*50)
    print(f"RESULTS REPORT")
    print("="*50)
    print(f"Mean Squared Error (MSE):   {mse:.5f}")
    print(f"Mean Absolute Error (MAE):  {mae:.5f}")
    print(f"R2 Score:                   {r2:.4f}")
    print("-" * 30)
    print(f"Avg Pixel Error:            ~{int(pixel_err_x)}px (X), ~{int(pixel_err_y)}px (Y)")
    print("-" * 30)
    print(f"Accuracy (within 5% screen):  {acc_5:.2f}%")
    print(f"Accuracy (within 10% screen): {acc_10:.2f}%")
    print("="*50)

    # Plot
    try:
        plt.figure(figsize=(10, 6))
        limit = min(200, len(y_test))
        plt.scatter(y_test[:limit, 0], y_test[:limit, 1], c='green', label='Actual', alpha=0.6)
        plt.scatter(y_pred[:limit, 0], y_pred[:limit, 1], c='red', marker='x', label='Predicted', alpha=0.6)
        for i in range(limit):
            plt.plot([y_test[i,0], y_pred[i,0]], [y_test[i,1], y_pred[i,1]], 'gray', alpha=0.2)
        plt.legend()
        plt.title(f"Gaze Prediction (First {limit} samples)")
        plt.gca().invert_yaxis()
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()