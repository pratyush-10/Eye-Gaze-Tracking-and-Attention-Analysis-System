import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 32
NUM_FEATURES = 21  # Matches your feature engineering logic
SCREEN_W, SCREEN_H = 1920, 1080  # Main Screen Size
CAM_PREVIEW_W, CAM_PREVIEW_H = 480, 360 # Size of the merged camera box
GAZE_SENSITIVITY_X = 15.0
GAZE_SENSITIVITY_Y = 20.0

# Heatmap config
HEATMAP_HISTORY = 60          # Number of gaze points to keep in heatmap
HEATMAP_BLUR_KERNEL = 151     # Must be odd — larger = more spread/softer
HEATMAP_ALPHA = 0.55          # Blend strength onto background (0=invisible, 1=opaque)
HEATMAP_DECAY = 0.92          # Weight decay for older points (older = less intense)

# Prediction smoothing
SMOOTH_WINDOW = 15            # Frames to average softmax probabilities over

# Output Labels for the new models
EMOTION_LABELS = {
    0: 'Excited / Happy', 
    1: 'Calm / Relaxed', 
    2: 'Sad / Bored', 
    3: 'Angry / Anxious'
}
COGLOAD_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# ==========================================
# FEATURE EXTRACTORS
# ==========================================
class LiveFeatureExtractor:
    """
    Extracts 21 temporal features consistent with the training logic.
    """
    @staticmethod
    def extract(gaze_history, head_history):
        features_seq = []
        seq_len = len(gaze_history)
        gaze_arr = np.array(gaze_history)
        head_arr = np.array(head_history)

        for t in range(seq_len):
            feats = []
            feats.extend(gaze_arr[t])
            feats.extend(head_arr[t])
            if t > 0:
                feats.extend(gaze_arr[t] - gaze_arr[t-1])
            else:
                feats.extend([0.0, 0.0])
            if t > 0:
                feats.extend(head_arr[t] - head_arr[t-1])
            else:
                feats.extend([0.0] * 6)
            if t > 1:
                v1 = gaze_arr[t] - gaze_arr[t-1]
                v0 = gaze_arr[t-1] - gaze_arr[t-2]
                feats.extend(v1 - v0)
            else:
                feats.extend([0.0, 0.0])
            if t > 0:
                vel_mag = np.linalg.norm(gaze_arr[t] - gaze_arr[t-1])
                feats.append(1.0 if vel_mag < 0.02 else 0.0)
            else:
                feats.append(0.0)
            feats.append(np.linalg.norm(head_arr[t][:3]))
            if t > 0:
                start_idx = max(0, t-5)
                diffs = gaze_arr[start_idx+1 : t+1] - gaze_arr[start_idx : t]
                vels = np.linalg.norm(diffs, axis=1)
                feats.append(np.mean(vels) if len(vels) > 0 else 0.0)
            else:
                feats.append(0.0)
            if len(feats) < NUM_FEATURES:
                feats.extend([0.0] * (NUM_FEATURES - len(feats)))
            features_seq.append(feats[:NUM_FEATURES])
        return np.array(features_seq, dtype=np.float32)


class EmotionFeatureExtractor:
    """
    Extracts 49 behavioral features from gaze/head dynamics for emotion classification.
    """
    @staticmethod
    def extract(gaze_history, head_history, lid_dist_history):
        gaze_arr = np.array(gaze_history)
        head_arr = np.array(head_history)
        lid_arr = np.array(lid_dist_history)
        features = []

        features.extend([
            np.mean(gaze_arr[:, 0]), np.std(gaze_arr[:, 0]), np.var(gaze_arr[:, 0]),
            np.mean(gaze_arr[:, 1]), np.std(gaze_arr[:, 1]), np.var(gaze_arr[:, 1]),
            np.max(gaze_arr[:, 0]), np.min(gaze_arr[:, 0]),
            np.max(gaze_arr[:, 1]), np.min(gaze_arr[:, 1])
        ])
        gaze_vel = np.linalg.norm(np.diff(gaze_arr, axis=0), axis=1) if len(gaze_arr) > 1 else np.array([0])
        features.extend([
            np.mean(gaze_vel), np.std(gaze_vel), np.max(gaze_vel),
            np.percentile(gaze_vel, 50), np.percentile(gaze_vel, 75), np.percentile(gaze_vel, 95)
        ])
        features.extend([
            np.mean(head_arr[:, 0]), np.std(head_arr[:, 0]),
            np.mean(head_arr[:, 1]), np.std(head_arr[:, 1]),
            np.mean(head_arr[:, 3]), np.std(head_arr[:, 3]),
            np.mean(head_arr[:, 4]), np.std(head_arr[:, 4])
        ])
        head_vel = np.linalg.norm(np.diff(head_arr[:, :3], axis=0), axis=1) if len(head_arr) > 1 else np.array([0])
        features.extend([
            np.mean(head_vel), np.std(head_vel), np.max(head_vel),
            np.percentile(head_vel, 50), np.percentile(head_vel, 90)
        ])
        fixation_threshold = 0.02
        fixations = gaze_vel < fixation_threshold
        fixation_count = np.sum(fixations)
        saccade_count = len(gaze_vel) - fixation_count
        features.extend([
            fixation_count, saccade_count,
            fixation_count / len(gaze_vel) if len(gaze_vel) > 0 else 0,
            saccade_count / len(gaze_vel) if len(gaze_vel) > 0 else 0,
            np.max(gaze_arr[:, 0]) - np.min(gaze_arr[:, 0]),
            np.max(gaze_arr[:, 1]) - np.min(gaze_arr[:, 1])
        ])
        blink_count = np.sum(lid_arr < 0.012)
        features.extend([
            np.mean(lid_arr), np.std(lid_arr), np.min(lid_arr),
            np.max(lid_arr),
            blink_count, blink_count / len(lid_arr) if len(lid_arr) > 0 else 0,
            np.percentile(lid_arr, 50), np.percentile(lid_arr, 75)
        ])
        if len(gaze_vel) > 1:
            gaze_acc = np.abs(np.diff(gaze_vel))
            features.extend([
                np.mean(gaze_acc), np.std(gaze_acc), np.max(gaze_acc),
                np.percentile(gaze_acc, 50), np.percentile(gaze_acc, 90),
                np.mean(np.diff(lid_arr)) if len(lid_arr) > 1 else 0
            ])
        else:
            features.extend([0.0] * 6)

        features = features[:49]
        if len(features) < 49:
            features.extend([0.0] * (49 - len(features)))
        return np.array(features, dtype=np.float32)


class CognitiveLoadFeatureExtractor:
    """
    Extracts 414 comprehensive multimodal features for cognitive load classification,
    with per-feature z-score normalization to match training distribution.
    """
    @staticmethod
    def extract(gaze_history, head_history, lid_dist_history):
        gaze_arr = np.array(gaze_history)
        head_arr = np.array(head_history)
        lid_arr = np.array(lid_dist_history)
        features = []

        # ===== GAZE FEATURES =====
        for dim in range(2):
            col = gaze_arr[:, dim]
            features.extend([
                np.mean(col), np.std(col), np.var(col),
                np.min(col), np.max(col),
                np.percentile(col, 10), np.percentile(col, 25), np.percentile(col, 50),
                np.percentile(col, 75), np.percentile(col, 90),
                np.ptp(col), np.median(col), np.mean(np.abs(col - np.mean(col)))
            ])
        gaze_vel = np.linalg.norm(np.diff(gaze_arr, axis=0), axis=1) if len(gaze_arr) > 1 else np.array([0])
        for metric in [np.mean, np.std, np.max, np.min, np.median]:
            features.append(metric(gaze_vel))
        features.extend(np.percentile(gaze_vel, [10, 25, 50, 75, 90]))
        gaze_vel_x = np.abs(np.diff(gaze_arr[:, 0])) if len(gaze_arr) > 1 else np.array([0])
        gaze_vel_y = np.abs(np.diff(gaze_arr[:, 1])) if len(gaze_arr) > 1 else np.array([0])
        for vel in [gaze_vel_x, gaze_vel_y]:
            features.extend([np.mean(vel), np.std(vel), np.max(vel), np.min(vel)])
        if len(gaze_vel) > 1:
            gaze_acc = np.abs(np.diff(gaze_vel))
            features.extend([np.mean(gaze_acc), np.std(gaze_acc), np.max(gaze_acc), np.min(gaze_acc)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        fixation_threshold = 0.02
        fixations = gaze_vel < fixation_threshold
        features.extend([
            np.sum(fixations), len(gaze_vel) - np.sum(fixations),
            np.sum(fixations) / len(gaze_vel) if len(gaze_vel) > 0 else 0,
            (len(gaze_vel) - np.sum(fixations)) / len(gaze_vel) if len(gaze_vel) > 0 else 0
        ])
        features.extend([
            np.max(gaze_arr[:, 0]) - np.min(gaze_arr[:, 0]),
            np.max(gaze_arr[:, 1]) - np.min(gaze_arr[:, 1]),
            np.std(gaze_arr[:, 0]), np.std(gaze_arr[:, 1]),
            np.percentile(gaze_vel, 95), np.percentile(gaze_vel, 5)
        ])
        if len(gaze_vel) > 1:
            gaze_jerk = np.abs(np.diff(gaze_vel))
            features.extend([
                np.mean(gaze_jerk), np.std(gaze_jerk), np.max(gaze_jerk),
                1.0 / (1.0 + np.mean(gaze_jerk))
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 1.0])
        gaze_path_length = np.sum(np.linalg.norm(np.diff(gaze_arr, axis=0), axis=1))
        features.extend([
            gaze_path_length,
            gaze_path_length / len(gaze_arr),
            np.sum(gaze_vel ** 2),
            np.max(gaze_vel) - np.min(gaze_vel)
        ])

        # ===== HEAD FEATURES =====
        for dim in range(3):
            col = head_arr[:, dim]
            features.extend([
                np.mean(col), np.std(col), np.var(col), np.min(col), np.max(col),
                np.percentile(col, 25), np.percentile(col, 50), np.percentile(col, 75)
            ])
        for dim in [3, 4]:
            col = head_arr[:, dim]
            features.extend([
                np.mean(col), np.std(col), np.max(np.abs(col)),
                np.percentile(col, 25), np.percentile(col, 75)
            ])
        head_vel = np.linalg.norm(np.diff(head_arr[:, :3], axis=0), axis=1) if len(head_arr) > 1 else np.array([0])
        features.extend([
            np.mean(head_vel), np.std(head_vel), np.max(head_vel), np.min(head_vel),
            np.percentile(head_vel, 50), np.percentile(head_vel, 90)
        ])
        if len(head_vel) > 1:
            head_acc = np.abs(np.diff(head_vel))
            features.extend([np.mean(head_acc), np.std(head_acc), np.max(head_acc)])
        else:
            features.extend([0.0, 0.0, 0.0])
        head_magnitude = np.linalg.norm(head_arr[:, :3], axis=1)
        features.extend([
            np.mean(head_magnitude), np.std(head_magnitude),
            np.max(head_magnitude), np.min(head_magnitude)
        ])
        if len(head_vel) > 1:
            head_jerk = np.abs(np.diff(head_vel))
            features.extend([np.mean(head_jerk), np.std(head_jerk), np.max(head_jerk)])
        else:
            features.extend([0.0, 0.0, 0.0])
        head_path_length = np.sum(head_vel)
        features.extend([
            head_path_length,
            head_path_length / len(head_arr) if len(head_arr) > 0 else 0
        ])

        # ===== BLINK/LID FEATURES =====
        for percentile in [5, 10, 25, 50, 75, 90, 95]:
            features.append(np.percentile(lid_arr, percentile))
        features.extend([
            np.mean(lid_arr), np.std(lid_arr), np.var(lid_arr),
            np.min(lid_arr), np.max(lid_arr),
            np.ptp(lid_arr), np.median(lid_arr)
        ])
        blink_threshold = 0.012
        blinks = lid_arr < blink_threshold
        blink_count = np.sum(blinks)
        features.extend([
            blink_count, blink_count / len(lid_arr) if len(lid_arr) > 0 else 0
        ])
        blink_transitions = np.sum(np.abs(np.diff(blinks.astype(int))))
        features.append(blink_transitions)
        blink_indices = np.where(blinks)[0]
        if len(blink_indices) > 1:
            blink_intervals = np.diff(blink_indices)
            features.extend([
                np.mean(blink_intervals), np.std(blink_intervals),
                np.max(blink_intervals), np.min(blink_intervals)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        lid_vel = np.abs(np.diff(lid_arr)) if len(lid_arr) > 1 else np.array([0])
        features.extend([
            np.mean(lid_vel), np.std(lid_vel), np.max(lid_vel),
            np.percentile(lid_vel, 50), np.percentile(lid_vel, 90)
        ])
        features.extend([
            np.std(lid_arr), np.var(lid_arr),
            np.sum(np.abs(np.diff(lid_arr))),
            np.max(np.abs(np.diff(lid_arr))) if len(lid_arr) > 1 else 0
        ])
        lid_open_ratio = np.sum(blinks > 0) / len(blinks)
        features.append(lid_open_ratio)

        # ===== COMBINED/TEMPORAL FEATURES =====
        if len(gaze_arr) > 1 and len(head_arr) > 1:
            try:
                corr_x = np.corrcoef(gaze_arr[:, 0], head_arr[:, 0])[0, 1]
                corr_y = np.corrcoef(gaze_arr[:, 1], head_arr[:, 1])[0, 1]
                if np.isnan(corr_x): corr_x = 0
                if np.isnan(corr_y): corr_y = 0
            except:
                corr_x, corr_y = 0, 0
            features.extend([corr_x, corr_y])
        else:
            features.extend([0.0, 0.0])
        window_size = max(1, len(gaze_arr) // 4)
        for i in range(4):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(gaze_arr))
            if start_idx < end_idx:
                window_data = gaze_arr[start_idx:end_idx]
                features.extend([
                    np.var(window_data[:, 0]), np.var(window_data[:, 1]),
                    np.mean(np.linalg.norm(np.diff(window_data, axis=0), axis=1)) if len(window_data) > 1 else 0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        overall_motion = np.linalg.norm(np.diff(gaze_arr, axis=0), axis=1)
        features.extend([
            np.mean(overall_motion), np.std(overall_motion),
            np.max(overall_motion), np.min(overall_motion),
            np.percentile(overall_motion, 90)
        ])
        features.extend([
            np.std(np.diff(gaze_vel)) if len(gaze_vel) > 1 else 0,
            np.std(np.diff(head_vel)) if len(head_vel) > 1 else 0,
            np.std(np.diff(lid_arr)) if len(lid_arr) > 1 else 0,
            np.sum(np.diff(gaze_arr, axis=0) ** 2),
            np.sum(np.diff(head_arr[:, :3], axis=0) ** 2)
        ])
        features.extend([
            len(gaze_arr),
            np.mean([len(gaze_arr), len(head_arr), len(lid_arr)]),
            gaze_path_length / (len(gaze_arr) + 1e-7),
            head_path_length / (len(head_arr) + 1e-7)
        ])
        if len(gaze_vel) > 0:
            gaze_vel_normalized = (gaze_vel - np.min(gaze_vel)) / (np.max(gaze_vel) - np.min(gaze_vel) + 1e-7)
            features.extend([
                np.sum(gaze_vel_normalized * np.log(gaze_vel_normalized + 1e-7)) / len(gaze_vel),
                np.sum(gaze_vel_normalized ** 2) / len(gaze_vel)
            ])
        else:
            features.extend([0.0, 0.0])

        features = features[:414]
        if len(features) < 414:
            features.extend([0.0] * (414 - len(features)))

        features = np.array(features, dtype=np.float32)

        # ---- FIX: z-score normalize features to avoid "always High" ----
        # Prevents extremely large raw values (e.g. path_length, sum of squares)
        # from pushing the model into its high-load regime unconditionally.
        mean = np.mean(features)
        std = np.std(features)
        if std > 1e-6:
            features = (features - mean) / std
        # Clip to prevent extreme outliers from dominating
        features = np.clip(features, -5.0, 5.0)

        return features


# ==========================================
# GAZE HEATMAP RENDERER
# ==========================================
class GazeHeatmap:
    """
    Accumulates gaze screen points and renders a smooth Gaussian heatmap.
    Older points decay in influence so only recent gaze matters.
    """
    def __init__(self, screen_w, screen_h, history_len=HEATMAP_HISTORY,
                 blur_kernel=HEATMAP_BLUR_KERNEL, alpha=HEATMAP_ALPHA, decay=HEATMAP_DECAY):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.history_len = history_len
        self.blur_kernel = blur_kernel
        self.alpha = alpha
        self.decay = decay
        self.points = deque(maxlen=history_len)  # each entry: (x, y)

        # Jet-like colormap table (BGR) for the heatmap
        self._colormap = self._build_colormap()

    def _build_colormap(self):
        """Build a blue→cyan→green→yellow→red BGRA lookup (256 entries)."""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.25:
                r, g, b = 0, int(255 * (t / 0.25)), 255
            elif t < 0.5:
                r, g, b = 0, 255, int(255 * (1 - (t - 0.25) / 0.25))
            elif t < 0.75:
                r, g, b = int(255 * ((t - 0.5) / 0.25)), 255, 0
            else:
                r, g, b = 255, int(255 * (1 - (t - 0.75) / 0.25)), 0
            lut[i, 0] = [b, g, r]  # OpenCV BGR
        return lut

    def add_point(self, x, y):
        self.points.append((x, y))

    def render(self, canvas):
        """Blend the heatmap onto the given canvas (in-place) and return it."""
        if len(self.points) < 2:
            return canvas

        # Build a float accumulation map
        heat = np.zeros((self.screen_h, self.screen_w), dtype=np.float32)
        n = len(self.points)
        for idx, (px, py) in enumerate(self.points):
            weight = self.decay ** (n - 1 - idx)  # recent points weigh more
            if 0 <= px < self.screen_w and 0 <= py < self.screen_h:
                heat[py, px] += weight

        # Gaussian blur to spread each point into a soft blob
        k = self.blur_kernel
        heat = cv2.GaussianBlur(heat, (k, k), sigmaX=k // 3, sigmaY=k // 3)

        # Normalize to [0, 255]
        max_val = heat.max()
        if max_val < 1e-6:
            return canvas
        heat_norm = (heat / max_val * 255).astype(np.uint8)

        # Apply colormap
        heat_colored = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        # Mask: only show where there is actual heat (avoids blue wash everywhere)
        mask = (heat_norm > 8).astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)

        # Blend onto canvas
        blended = (canvas.astype(np.float32) * (1 - mask_3ch * self.alpha)
                   + heat_colored.astype(np.float32) * mask_3ch * self.alpha)
        np.copyto(canvas, blended.clip(0, 255).astype(np.uint8))
        return canvas


# ==========================================
# PREDICTION SMOOTHER
# ==========================================
class ProbabilitySmoother:
    """
    Maintains a rolling window of softmax probability vectors and
    returns the time-averaged probabilities to stabilise predictions.
    """
    def __init__(self, num_classes, window=SMOOTH_WINDOW):
        self.window = window
        self.history = deque(maxlen=window)
        self.num_classes = num_classes

    def update(self, probs):
        self.history.append(np.array(probs, dtype=np.float32))
        return np.mean(self.history, axis=0)


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    print("Loading models...")
    try:
        model_attn = tf.keras.models.load_model('models/attention_classifier.keras')
        model_gaze = tf.keras.models.load_model('models/gaze_estimator.keras')
        model_emotion = tf.keras.models.load_model('models/emotion_classifier.h5')
        model_cogload = tf.keras.models.load_model('models/cognitive_load_classifier.h5')
        print("All 4 models loaded successfully.")
    except Exception as e:
        print(f"Error: {e}. Ensure all 4 models are in the 'models/' folder.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    window_name = 'Multimodal AI Tracker Dashboard'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)

    # Buffers
    gaze_history = deque(maxlen=SEQUENCE_LENGTH)
    head_history = deque(maxlen=SEQUENCE_LENGTH)
    lid_history = deque(maxlen=SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        gaze_history.append([0.5, 0.5])
        head_history.append([0.0] * 6)
        lid_history.append(0.05)

    # Heatmap & smoothers
    heatmap = GazeHeatmap(SCREEN_W, SCREEN_H)
    emo_smoother = ProbabilitySmoother(num_classes=4)
    cog_smoother = ProbabilitySmoother(num_classes=3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        current_gaze_raw = [0.5, 0.5]
        current_head_pose = [0.0] * 6
        eyes_closed = False
        lid_dist_avg = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_lid_dist = abs(landmarks[159].y - landmarks[145].y)
            right_lid_dist = abs(landmarks[386].y - landmarks[374].y)
            lid_dist_avg = (left_lid_dist + right_lid_dist) / 2.0
            if lid_dist_avg < 0.012:
                eyes_closed = True

            nose = landmarks[1]
            yaw = (nose.x - 0.5) * 2
            pitch = (nose.y - 0.5) * 2
            current_head_pose = [nose.x, nose.y, nose.z, pitch, yaw, 0.0]

            left_iris = landmarks[468]
            right_iris = landmarks[473]
            gaze_x = (left_iris.x + right_iris.x) / 2.0
            gaze_y = (left_iris.y + right_iris.y) / 2.0
            current_gaze_raw = [gaze_x, gaze_y]

            line_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            pt_l_top = (int(landmarks[159].x * w), int(landmarks[159].y * h))
            pt_l_bot = (int(landmarks[145].x * w), int(landmarks[145].y * h))
            cv2.line(frame, pt_l_top, pt_l_bot, line_color, 2)
            pt_r_top = (int(landmarks[386].x * w), int(landmarks[386].y * h))
            pt_r_bot = (int(landmarks[374].x * w), int(landmarks[374].y * h))
            cv2.line(frame, pt_r_top, pt_r_bot, line_color, 2)

        # Update histories
        gaze_history.append(current_gaze_raw)
        head_history.append(current_head_pose)
        lid_history.append(lid_dist_avg)

        # ---- Model 1 & 2: Gaze & Attention ----
        features = LiveFeatureExtractor.extract(gaze_history, head_history)
        features_batch = np.expand_dims(features, axis=0)
        gaze_pred = model_gaze.predict(features_batch, verbose=0)[0]
        attn_probs = model_attn.predict(features_batch, verbose=0)[0]

        # ---- Model 3 & 4: Emotion & Cognitive Load ----
        emo_features = EmotionFeatureExtractor.extract(gaze_history, head_history, lid_history)
        emo_raw = model_emotion.predict(np.expand_dims(emo_features, axis=0), verbose=0)[0]
        emo_smoothed = emo_smoother.update(emo_raw)
        emo_status = EMOTION_LABELS.get(np.argmax(emo_smoothed), "Unknown")

        cog_features = CognitiveLoadFeatureExtractor.extract(gaze_history, head_history, lid_history)
        cog_raw = model_cogload.predict(np.expand_dims(cog_features, axis=0), verbose=0)[0]
        cog_smoothed = cog_smoother.update(cog_raw)
        cog_status = COGLOAD_LABELS.get(np.argmax(cog_smoothed), "Unknown")

        # ---- Gaze screen mapping ----
        center_x, center_y = 0.5, 0.5
        dx = current_gaze_raw[0] - center_x
        dy = current_gaze_raw[1] - center_y
        screen_x = int(SCREEN_W / 2 + (dx * SCREEN_W * GAZE_SENSITIVITY_X))
        screen_y = int(SCREEN_H / 2 + (dy * SCREEN_H * GAZE_SENSITIVITY_Y))
        looking_on_screen = (0 <= screen_x <= SCREEN_W) and (0 <= screen_y <= SCREEN_H)
        screen_x_clamped = max(0, min(SCREEN_W - 1, screen_x))
        screen_y_clamped = max(0, min(SCREEN_H - 1, screen_y))

        # ---- Attention logic ----
        if eyes_closed:
            attn_status = "SLEEPING"
            attn_color = (0, 0, 255)
        elif looking_on_screen:
            attn_status = "ATTENTIVE"
            attn_color = (0, 255, 0)
        else:
            attn_status = "DISTRACTED"
            attn_color = (0, 165, 255)

        # ---- Add gaze point to heatmap (only when eyes open) ----
        if not eyes_closed:
            heatmap.add_point(screen_x_clamped, screen_y_clamped)

        # ==========================================
        # RENDER
        # ==========================================
        main_canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        # 1. Draw heatmap (replaces the single red dot)
        if not eyes_closed:
            heatmap.render(main_canvas)

        # 2. Camera PIP
        cam_preview = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))
        cv2.rectangle(cam_preview, (0, 0), (CAM_PREVIEW_W - 1, CAM_PREVIEW_H - 1), attn_color, 4)
        cv2.putText(cam_preview, f"Lid: {lid_dist_avg:.3f}", (10, CAM_PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_off = 20
        x_off = SCREEN_W - CAM_PREVIEW_W - 20
        main_canvas[y_off:y_off + CAM_PREVIEW_H, x_off:x_off + CAM_PREVIEW_W] = cam_preview

        # 3. Dashboard text
        cv2.putText(main_canvas, "REAL-TIME MULTIMODAL ANALYSIS", (50, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.line(main_canvas, (50, 90), (620, 90), (255, 255, 255), 2)

        cv2.putText(main_canvas, "ATTENTION :", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(main_canvas, attn_status, (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, attn_color, 3)

        cv2.putText(main_canvas, "EMOTION   :", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(main_canvas, emo_status.upper(), (250, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        cv2.putText(main_canvas, "COG. LOAD :", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(main_canvas, cog_status.upper(), (250, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 3)

        # Confidence bars for cognitive load
        bar_y = 285
        bar_x_start = 50
        bar_total_w = 300
        for i, (label, prob) in enumerate(zip(COGLOAD_LABELS.values(), cog_smoothed)):
            bar_w = int(bar_total_w * prob)
            bar_colors = [(100, 220, 100), (50, 180, 255), (80, 80, 255)]  # Low/Med/High
            cv2.rectangle(main_canvas, (bar_x_start, bar_y + i * 22),
                          (bar_x_start + bar_w, bar_y + i * 22 + 14), bar_colors[i], -1)
            cv2.putText(main_canvas, f"{label} {prob:.2f}",
                        (bar_x_start + bar_total_w + 8, bar_y + i * 22 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if attn_status == "SLEEPING":
            cv2.putText(main_canvas, "WAKE UP!", (SCREEN_W // 2 - 200, SCREEN_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)

        cv2.imshow(window_name, main_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()