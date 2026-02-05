import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCREEN_W, SCREEN_H = 1920, 1080
CAM_PREVIEW_W, CAM_PREVIEW_H = 480, 360

# Stabilization Parameters
MIN_CUTOFF = 0.005  # Lower = More smoothing (less jitter)
BETA = 0.05         # Higher = More responsiveness (less lag)

# Calibration Points (9-point grid)
CALIB_POINTS = [
    (50, 50), (SCREEN_W//2, 50), (SCREEN_W-50, 50),
    (50, SCREEN_H//2), (SCREEN_W//2, SCREEN_H//2), (SCREEN_W-50, SCREEN_H//2),
    (50, SCREEN_H-50), (SCREEN_W//2, SCREEN_H-50), (SCREEN_W-50, SCREEN_H-50)
]

# ==========================================
# 2. STABILIZER (1€ Filter)
# ==========================================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, t, x):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

# ==========================================
# 3. FEATURE EXTRACTOR (For AI Models)
# ==========================================
class LiveFeatureExtractor:
    @staticmethod
    def extract(gaze_history, head_history):
        # Replicates your training feature engineering
        features_seq = []
        seq_len = len(gaze_history)
        gaze_arr = np.array(gaze_history)
        head_arr = np.array(head_history)
        num_features = 21

        for t in range(seq_len):
            feats = []
            feats.extend(gaze_arr[t]) # Gaze (2)
            feats.extend(head_arr[t]) # Head (6)
            
            # Velocities
            if t > 0:
                feats.extend(gaze_arr[t] - gaze_arr[t-1])
                feats.extend(head_arr[t] - head_arr[t-1])
            else:
                feats.extend([0.0]*8)
                
            # Acceleration
            if t > 1:
                v1 = gaze_arr[t] - gaze_arr[t-1]
                v0 = gaze_arr[t-1] - gaze_arr[t-2]
                feats.extend(v1 - v0)
            else:
                feats.extend([0.0, 0.0])

            # Fixation & Magnitude
            if t > 0:
                mag = np.linalg.norm(gaze_arr[t] - gaze_arr[t-1])
                feats.append(1.0 if mag < 0.02 else 0.0)
            else:
                feats.append(0.0)
            
            feats.append(np.linalg.norm(head_arr[t][:3]))
            feats.append(0.0) # Placeholder for mean vel
            
            # Pad to 21
            if len(feats) < num_features:
                feats.extend([0.0]*(num_features-len(feats)))
            features_seq.append(feats[:num_features])
            
        return np.array(features_seq, dtype=np.float32)

# ==========================================
# 4. MAIN SYSTEM
# ==========================================
class EyeTrackingSystem:
    def __init__(self):
        # A. Setup MediaPipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, max_num_faces=1, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        # B. Load Models
        print("Loading AI Models...")
        try:
            self.model_attn = tf.keras.models.load_model('models/attention_model.keras')
            print("Attention Model Loaded.")
        except:
            print("WARNING: 'attention_model.keras' not found. Mocking predictions.")
            self.model_attn = None

        # C. Calibration & State
        self.calib_mode = True
        self.calib_step = 0
        self.calib_frame_counter = 0
        self.calib_data_x = [] # Eye vectors
        self.calib_data_y = [] # Screen coords
        
        self.poly_x = None
        self.poly_y = None
        
        # D. Filters & History
        self.filter_x = OneEuroFilter(MIN_CUTOFF, BETA)
        self.filter_y = OneEuroFilter(MIN_CUTOFF, BETA)
        
        self.gaze_history = deque(maxlen=32)
        self.head_history = deque(maxlen=32)
        for _ in range(32):
            self.gaze_history.append([0.5, 0.5])
            self.head_history.append([0.0]*6)
            
        self.last_valid_pos = (SCREEN_W//2, SCREEN_H//2)

    def get_eye_vector_and_ear(self, landmarks, w, h):
        """ Returns normalized eye vector (for tracking) and EAR (for closure) """
        # Left Eye Indices
        l_top, l_bot = landmarks[159], landmarks[145]
        l_in, l_out = landmarks[33], landmarks[133]
        l_iris = landmarks[468]
        
        # Right Eye Indices
        r_top, r_bot = landmarks[386], landmarks[374]
        r_in, r_out = landmarks[362], landmarks[263]
        r_iris = landmarks[473]

        # 1. Calculate Eye Aspect Ratio (EAR) - Distance between lids
        l_dist = abs(l_top.y - l_bot.y)
        r_dist = abs(r_top.y - r_bot.y)
        avg_ear = (l_dist + r_dist) / 2.0
        
        # 2. Calculate Iris Vector relative to Eye Corners (Pupil Tracking)
        def rel_pos(inner, outer, iris):
            width = np.linalg.norm(np.array([outer.x, outer.y]) - np.array([inner.x, inner.y]))
            if width == 0: return 0.5, 0.5
            cx = (inner.x + outer.x)/2
            cy = (inner.y + outer.y)/2
            # Vector from center to iris
            dx = (iris.x - cx) / width
            dy = (iris.y - cy) / width
            return dx, dy

        lx, ly = rel_pos(l_in, l_out, l_iris)
        rx, ry = rel_pos(r_in, r_out, r_iris)
        
        # Average vector
        eye_vec = [(lx+rx)/2, (ly+ry)/2]
        
        # Iris Pixels for visualization
        l_iris_px = (int(l_iris.x * w), int(l_iris.y * h))
        r_iris_px = (int(r_iris.x * w), int(r_iris.y * h))
        
        # Lid Pixels for visualization
        lid_pts = [
            (int(l_top.x*w), int(l_top.y*h)), (int(l_bot.x*w), int(l_bot.y*h)),
            (int(r_top.x*w), int(r_top.y*h)), (int(r_bot.x*w), int(r_bot.y*h))
        ]
        
        return eye_vec, avg_ear, l_iris_px, r_iris_px, lid_pts

    def calibrate_math(self):
        # 2nd Degree Polynomial Regression
        X = np.array(self.calib_data_x)
        Y = np.array(self.calib_data_y)
        
        # Features: [x, y, x^2, y^2, xy]
        X_poly = np.c_[X, X**2, X[:,0]*X[:,1]] 
        A = np.hstack([X_poly, np.ones((X_poly.shape[0], 1))])
        
        self.poly_x, _, _, _ = np.linalg.lstsq(A, Y[:, 0], rcond=None)
        self.poly_y, _, _, _ = np.linalg.lstsq(A, Y[:, 1], rcond=None)
        print("Calibration Matrix Calculated.")

    def run(self):
        cap = cv2.VideoCapture(0)
        win_name = "Live Eye Tracking & Attention"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            # -----------------------------------
            # DATA EXTRACTION
            # -----------------------------------
            eye_vec = [0.0, 0.0]
            ear = 0.0
            l_iris_px, r_iris_px = (0,0), (0,0)
            lid_pts = []
            head_pose = [0.0]*6
            eyes_closed = False

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                eye_vec, ear, l_iris_px, r_iris_px, lid_pts = self.get_eye_vector_and_ear(lms, w, h)
                
                # Head Pose Approx
                nose = lms[1]
                yaw = (nose.x - 0.5) * 2
                pitch = (nose.y - 0.5) * 2
                head_pose = [nose.x, nose.y, nose.z, pitch, yaw, 0.0]

                # Threshold for sleeping/blinking
                if ear < 0.012: eyes_closed = True

            # -----------------------------------
            # CALIBRATION LOGIC
            # -----------------------------------
            main_canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

            if self.calib_mode:
                if self.calib_step < len(CALIB_POINTS):
                    tx, ty = CALIB_POINTS[self.calib_step]
                    
                    # Yellow Target
                    cv2.circle(main_canvas, (tx, ty), 20, (0, 255, 255), -1)
                    cv2.putText(main_canvas, f"LOOK AT DOT {self.calib_step+1}/9", (tx-60, ty+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    
                    self.calib_frame_counter += 1
                    # Capture frames 10 to 40
                    if 10 < self.calib_frame_counter < 40 and not eyes_closed and results.multi_face_landmarks:
                        self.calib_data_x.append(eye_vec)
                        self.calib_data_y.append([tx, ty])
                    
                    if self.calib_frame_counter >= 45:
                        self.calib_step += 1
                        self.calib_frame_counter = 0
                else:
                    self.calibrate_math()
                    self.calib_mode = False
            
            # -----------------------------------
            # TRACKING & AI LOGIC
            # -----------------------------------
            screen_pt = self.last_valid_pos
            status = "CALIBRATING"
            status_color = (200, 200, 200)

            if not self.calib_mode and results.multi_face_landmarks:
                # 1. PREDICT COORDINATES
                # Math: [x, y, x^2, y^2, xy, 1]
                ex, ey = eye_vec
                vec = np.array([ex, ey, ex**2, ey**2, ex*ey, 1.0])
                sx = np.dot(vec, self.poly_x)
                sy = np.dot(vec, self.poly_y)
                
                # 2. FILTERING (Fix Fidgeting)
                # If eyes are closed (blink), DO NOT update filter (Freeze dot)
                if not eyes_closed:
                    curr_time = time.time()
                    sx = self.filter_x(curr_time, sx)
                    sy = self.filter_y(curr_time, sy)
                    self.last_valid_pos = (int(sx), int(sy))
                
                screen_pt = self.last_valid_pos

                # 3. AI PREDICTION (ATTENTION)
                # Update Buffers
                self.gaze_history.append(eye_vec) # Using raw vec for model consistency
                self.head_history.append(head_pose)
                
                if self.model_attn:
                    feats = LiveFeatureExtractor.extract(self.gaze_history, self.head_history)
                    probs = self.model_attn.predict(np.expand_dims(feats, axis=0), verbose=0)[0]
                    ai_class = np.argmax(probs) # 0: Focused, 1: Distracted, 2: Sleep
                else:
                    ai_class = 0

                # 4. HYBRID STATUS LOGIC
                # Logic: If Eyes Closed -> SLEEPING
                # Logic: If Eyes Open & Dot on Screen -> ATTENTIVE
                # Logic: If Eyes Open & Dot off Screen -> DISTRACTED
                
                on_screen = (0 <= screen_pt[0] <= SCREEN_W) and (0 <= screen_pt[1] <= SCREEN_H)
                
                if eyes_closed:
                    status = "SLEEPING"
                    status_color = (0, 0, 255) # Red
                elif on_screen:
                    status = "ATTENTIVE"
                    status_color = (0, 255, 0) # Green
                else:
                    status = "DISTRACTED"
                    status_color = (0, 165, 255) # Orange

            # -----------------------------------
            # RENDERING
            # -----------------------------------
            
            # A. Draw Gaze Dot (Only if not calibrating)
            if not self.calib_mode:
                cv2.circle(main_canvas, screen_pt, 15, (0, 0, 255), -1) # Simple Red Dot
                
                if status == "SLEEPING":
                    cv2.putText(main_canvas, "WAKE UP!", (SCREEN_W//2 - 200, SCREEN_H//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
            
            # B. Prepare Camera Preview (PiP)
            # Resize
            cam_view = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))
            
            # Draw Indicators on Camera View
            if results.multi_face_landmarks:
                # Eyelids (Red if closed, Green if open)
                lc = (0,0,255) if eyes_closed else (0,255,0)
                # Left Lid
                cv2.line(cam_view, 
                         (int(lid_pts[0][0]*CAM_PREVIEW_W/w), int(lid_pts[0][1]*CAM_PREVIEW_H/h)),
                         (int(lid_pts[1][0]*CAM_PREVIEW_W/w), int(lid_pts[1][1]*CAM_PREVIEW_H/h)), lc, 2)
                # Right Lid
                cv2.line(cam_view, 
                         (int(lid_pts[2][0]*CAM_PREVIEW_W/w), int(lid_pts[2][1]*CAM_PREVIEW_H/h)),
                         (int(lid_pts[3][0]*CAM_PREVIEW_W/w), int(lid_pts[3][1]*CAM_PREVIEW_H/h)), lc, 2)
                
                # Pupils (Yellow)
                cv2.circle(cam_view, (int(l_iris_px[0]*CAM_PREVIEW_W/w), int(l_iris_px[1]*CAM_PREVIEW_H/h)), 3, (0,255,255), -1)
                cv2.circle(cam_view, (int(r_iris_px[0]*CAM_PREVIEW_W/w), int(r_iris_px[1]*CAM_PREVIEW_H/h)), 3, (0,255,255), -1)

            # Border for status
            cv2.rectangle(cam_view, (0,0), (CAM_PREVIEW_W, CAM_PREVIEW_H), status_color, 4)
            cv2.putText(cam_view, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # C. Merge Camera into Bottom Right
            y_off = SCREEN_H - CAM_PREVIEW_H - 30
            x_off = SCREEN_W - CAM_PREVIEW_W - 30
            main_canvas[y_off:y_off+CAM_PREVIEW_H, x_off:x_off+CAM_PREVIEW_W] = cam_view
            
            cv2.imshow(win_name, main_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    EyeTrackingSystem().run()