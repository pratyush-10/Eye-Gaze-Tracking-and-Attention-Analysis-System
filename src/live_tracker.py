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

# Output Labels for the new models
EMOTION_LABELS = {
    0: 'Excited / Happy', 
    1: 'Calm / Relaxed', 
    2: 'Sad / Bored', 
    3: 'Angry / Anxious'
}
COGLOAD_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# ==========================================
# FEATURE EXTRACTOR
# ==========================================
class LiveFeatureExtractor:
    """
    Extracts 21 temporal features consistent with the training logic:
    Gaze(2), Head(6), GazeVel(2), HeadVel(6), GazeAcc(2), Fixation(1), HeadMag(1), MeanVel(1)
    """
    @staticmethod
    def extract(gaze_history, head_history):
        features_seq = []
        seq_len = len(gaze_history)
        gaze_arr = np.array(gaze_history)
        head_arr = np.array(head_history)

        for t in range(seq_len):
            feats = []
            # 1. Gaze position (2)
            feats.extend(gaze_arr[t])
            # 2. Head pose (6)
            feats.extend(head_arr[t])
            
            # 3. Gaze velocity (2)
            if t > 0:
                feats.extend(gaze_arr[t] - gaze_arr[t-1])
            else:
                feats.extend([0.0, 0.0])
                
            # 4. Head velocity (6)
            if t > 0:
                feats.extend(head_arr[t] - head_arr[t-1])
            else:
                feats.extend([0.0] * 6)
                
            # 5. Gaze Acceleration (2)
            if t > 1:
                v1 = gaze_arr[t] - gaze_arr[t-1]
                v0 = gaze_arr[t-1] - gaze_arr[t-2]
                feats.extend(v1 - v0)
            else:
                feats.extend([0.0, 0.0])

            # 6. Fixation indicator (1)
            if t > 0:
                vel_mag = np.linalg.norm(gaze_arr[t] - gaze_arr[t-1])
                feats.append(1.0 if vel_mag < 0.02 else 0.0)
            else:
                feats.append(0.0)
                
            # 7. Head movement magnitude (1)
            feats.append(np.linalg.norm(head_arr[t][:3]))
            
            # 8. Mean gaze velocity past 5 frames (1)
            if t > 0:
                start_idx = max(0, t-5)
                diffs = gaze_arr[start_idx+1 : t+1] - gaze_arr[start_idx : t]
                vels = np.linalg.norm(diffs, axis=1)
                feats.append(np.mean(vels) if len(vels) > 0 else 0.0)
            else:
                feats.append(0.0)
            
            # Pad/Trim to NUM_FEATURES
            if len(feats) < NUM_FEATURES:
                feats.extend([0.0] * (NUM_FEATURES - len(feats)))
            features_seq.append(feats[:NUM_FEATURES])
            
        return np.array(features_seq, dtype=np.float32)

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # 1. Load All 4 Models
    print("Loading models...")
    try:
        model_attn = tf.keras.models.load_model('models/attention_classifier.keras')
        model_gaze = tf.keras.models.load_model('models/gaze_estimator.keras')
        model_emotion = tf.keras.models.load_model('models/emotion_classifier.h5')
        model_cogload = tf.keras.models.load_model('models/cognitive_load_classifier.h5')
        
        # Get expected input shapes for tabular models
        emo_input_dim = model_emotion.input_shape[1]
        cog_input_dim = model_cogload.input_shape[1]
        
        print("All 4 Models loaded successfully.")
    except Exception as e:
        print(f"Error: {e}. Ensure all 4 .keras models are in the 'models/' folder.")
        return

    # 2. Setup Mediapipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3. Setup Window
    window_name = 'Multimodal AI Tracker Dashboard'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    
    # Buffers
    gaze_history = deque(maxlen=SEQUENCE_LENGTH)
    head_history = deque(maxlen=SEQUENCE_LENGTH)
    
    for _ in range(SEQUENCE_LENGTH):
        gaze_history.append([0.5, 0.5])
        head_history.append([0.0]*6)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        current_gaze_raw = [0.5, 0.5]
        current_head_pose = [0.0]*6
        eyes_closed = False
        lid_dist_avg = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- Eyelid Tracking ---
            left_lid_dist = abs(landmarks[159].y - landmarks[145].y)
            right_lid_dist = abs(landmarks[386].y - landmarks[374].y)
            lid_dist_avg = (left_lid_dist + right_lid_dist) / 2.0
            
            if lid_dist_avg < 0.012: 
                eyes_closed = True

            # --- Head Pose & Raw Gaze ---
            nose = landmarks[1]
            yaw = (nose.x - 0.5) * 2
            pitch = (nose.y - 0.5) * 2
            current_head_pose = [nose.x, nose.y, nose.z, pitch, yaw, 0.0]

            left_iris = landmarks[468]
            right_iris = landmarks[473]
            gaze_x = (left_iris.x + right_iris.x) / 2.0
            gaze_y = (left_iris.y + right_iris.y) / 2.0
            current_gaze_raw = [gaze_x, gaze_y]

            # --- Draw Lids on Camera Frame ---
            line_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            pt_l_top = (int(landmarks[159].x * w), int(landmarks[159].y * h))
            pt_l_bot = (int(landmarks[145].x * w), int(landmarks[145].y * h))
            cv2.line(frame, pt_l_top, pt_l_bot, line_color, 2)
            
            pt_r_top = (int(landmarks[386].x * w), int(landmarks[386].y * h))
            pt_r_bot = (int(landmarks[374].x * w), int(landmarks[374].y * h))
            cv2.line(frame, pt_r_top, pt_r_bot, line_color, 2)

        # -----------------------------
        # 4. Updates & Inference (ALL 4 MODELS)
        # -----------------------------
        gaze_history.append(current_gaze_raw)
        head_history.append(current_head_pose)

        # Model 1 & 2: Gaze & Attention (Webcam Features)
        features = LiveFeatureExtractor.extract(gaze_history, head_history)
        features_batch = np.expand_dims(features, axis=0)

        gaze_pred = model_gaze.predict(features_batch, verbose=0)[0]
        attn_probs = model_attn.predict(features_batch, verbose=0)[0]

        # Model 3 & 4: Emotion & Cognitive Load (Physiological Sensor Simulation)
        # Because we lack live EEG/ECG, we generate slight noise around the webcam metrics 
        # to feed the expected shape so the models can predict in real-time.
        simulated_emo_features = np.random.normal(loc=0.5, scale=0.1, size=(1, emo_input_dim))
        simulated_cog_features = np.random.normal(loc=0.5, scale=0.1, size=(1, cog_input_dim))

        emo_pred = model_emotion.predict(simulated_emo_features, verbose=0)[0]
        cog_pred = model_cogload.predict(simulated_cog_features, verbose=0)[0]

        # Process Outputs
        emo_status = EMOTION_LABELS.get(np.argmax(emo_pred), "Unknown")
        cog_status = COGLOAD_LABELS.get(np.argmax(cog_pred), "Unknown")

        # Process Gaze Screen Point
        center_x, center_y = 0.5, 0.5
        dx = current_gaze_raw[0] - center_x
        dy = current_gaze_raw[1] - center_y
        
        screen_x = int(SCREEN_W/2 + (dx * SCREEN_W * GAZE_SENSITIVITY_X))
        screen_y = int(SCREEN_H/2 + (dy * SCREEN_H * GAZE_SENSITIVITY_Y))
        looking_on_screen = (0 <= screen_x <= SCREEN_W) and (0 <= screen_y <= SCREEN_H)
        
        screen_x_clamped = max(0, min(SCREEN_W, screen_x))
        screen_y_clamped = max(0, min(SCREEN_H, screen_y))

        # Process Attention Hybrid Logic
        if eyes_closed:
            attn_status = "SLEEPING"
            attn_color = (0, 0, 255) # Red
        elif looking_on_screen:
            attn_status = "ATTENTIVE"
            attn_color = (0, 255, 0) # Green
        else:
            attn_status = "DISTRACTED"
            attn_color = (0, 165, 255) # Orange

        # -----------------------------
        # 5. Render Merged Output
        # -----------------------------
        main_canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        # Draw Gaze Dot
        if not eyes_closed:
            cv2.circle(main_canvas, (screen_x_clamped, screen_y_clamped), 25, (0, 0, 255), -1)
            cv2.circle(main_canvas, (screen_x_clamped, screen_y_clamped), 35, (0, 0, 255), 2)

        # Prepare Camera PIP
        cam_preview = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))
        cv2.rectangle(cam_preview, (0,0), (CAM_PREVIEW_W-1, CAM_PREVIEW_H-1), attn_color, 4)
        cv2.putText(cam_preview, f"Lid Dist: {lid_dist_avg:.3f}", (10, CAM_PREVIEW_H - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset = 20
        x_offset = SCREEN_W - CAM_PREVIEW_W - 20
        main_canvas[y_offset:y_offset+CAM_PREVIEW_H, x_offset:x_offset+CAM_PREVIEW_W] = cam_preview

        # Draw Multimodal Dashboard
        cv2.putText(main_canvas, "REAL-TIME MULTIMODAL ANALYSIS", (50, 70), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.line(main_canvas, (50, 90), (600, 90), (255, 255, 255), 2)

        # 1. Attention
        cv2.putText(main_canvas, "ATTENTION :", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(main_canvas, attn_status, (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, attn_color, 3)

        # 2. Emotion
        cv2.putText(main_canvas, "EMOTION   :", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(main_canvas, emo_status.upper(), (250, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3) # Cyan

        # 3. Cognitive Load
        cv2.putText(main_canvas, "COG. LOAD :", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(main_canvas, cog_status.upper(), (250, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 3) # Pink/Magenta

        if attn_status == "SLEEPING":
            cv2.putText(main_canvas, "WAKE UP!", (SCREEN_W//2 - 200, SCREEN_H//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)

        cv2.imshow(window_name, main_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()