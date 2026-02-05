"""
Real-Time Eye Tracking Demo for MPIIGaze Project - IMPROVED
With proper eyelid tracking and working gaze estimation
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import *


class ImprovedEyeTracker:
    """Real-time eye tracking with eyelid detection and working gaze"""
    
    def __init__(self):
        print("=" * 70)
        print("IMPROVED MPIIGAZE REAL-TIME EYE TRACKER")
        print("=" * 70)
        
        # Get screen dimensions
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        print(f"\nScreen: {self.screen_width}x{self.screen_height}")
        
        # Initialize MediaPipe Face Mesh
        print("\n[1/3] Loading MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe loaded")
        
        # Load trained models
        print("\n[2/3] Loading trained models...")
        try:
            self.attention_model = keras.models.load_model(
                str(MODELS_DIR / 'attention_model.keras')
            )
            print("✓ Attention model loaded")
        except Exception as e:
            print(f"⚠ Attention model: {e}")
            self.attention_model = None
        
        try:
            self.gaze_model = keras.models.load_model(
                str(MODELS_DIR / 'gaze_model.keras')
            )
            print("✓ Gaze model loaded")
        except Exception as e:
            print(f"⚠ Gaze model: {e}")
            self.gaze_model = None
        
        # Initialize webcam
        print("\n[3/3] Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        print("✓ Camera ready")
        
        # Buffers
        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # State
        self.gaze_point = np.array([self.screen_width // 2, self.screen_height // 2], dtype=float)
        self.attention_status = "Initializing..."
        self.attention_class = 0
        self.attention_confidence = 0.0
        
        # Eyelid tracking
        self.left_eye_open = True
        self.right_eye_open = True
        self.left_ear = 0.3
        self.right_ear = 0.3
        self.eyes_closed_frames = 0
        self.blink_threshold = 0.21  # Calibrated threshold
        
        # Performance
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # Smoothing
        self.gaze_history = deque(maxlen=5)
        
        # Debug
        self.debug_mode = True
        self.show_landmarks = True
        
        print("\n" + "=" * 70)
        print("CONTROLS:")
        print("  Q/ESC - Quit")
        print("  R - Reset")
        print("  D - Toggle debug")
        print("  L - Toggle landmarks")
        print("=" * 70 + "\n")
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate EAR for eyelid detection"""
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if h < 0.001:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def extract_comprehensive_features(self, landmarks, frame_shape):
        """Extract features with detailed eyelid tracking"""
        h, w = frame_shape[:2]
        
        # Define eye landmarks (MediaPipe indices)
        # Left eye: outer, top, inner, bottom, top-mid, bottom-mid
        LEFT_EYE = [33, 160, 133, 144, 159, 145]
        RIGHT_EYE = [362, 387, 263, 374, 386, 380]
        
        # Left eyelid landmarks (more detailed)
        LEFT_EYE_TOP = [159, 145, 158]  # Upper lid
        LEFT_EYE_BOTTOM = [145, 153, 154]  # Lower lid
        
        # Right eyelid landmarks
        RIGHT_EYE_TOP = [386, 374, 385]  # Upper lid
        RIGHT_EYE_BOTTOM = [374, 380, 381]  # Lower lid
        
        # Iris centers
        LEFT_IRIS = 468
        RIGHT_IRIS = 473
        
        # Extract left eye points
        left_eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                    for i in LEFT_EYE])
        right_eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                     for i in RIGHT_EYE])
        
        # Extract eyelid points
        left_top_lid = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                 for i in LEFT_EYE_TOP])
        left_bottom_lid = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                    for i in LEFT_EYE_BOTTOM])
        right_top_lid = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                  for i in RIGHT_EYE_TOP])
        right_bottom_lid = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                     for i in RIGHT_EYE_BOTTOM])
        
        # Eye centers
        left_eye_center = left_eye_points.mean(axis=0)
        right_eye_center = right_eye_points.mean(axis=0)
        
        # Iris positions
        if len(landmarks) > RIGHT_IRIS:
            left_iris = np.array([landmarks[LEFT_IRIS].x * w, landmarks[LEFT_IRIS].y * h])
            right_iris = np.array([landmarks[RIGHT_IRIS].x * w, landmarks[RIGHT_IRIS].y * h])
        else:
            left_iris = left_eye_center
            right_iris = right_eye_center
        
        # Calculate EAR for both eyes
        self.left_ear = self.calculate_eye_aspect_ratio(
            np.array([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE])
        )
        self.right_ear = self.calculate_eye_aspect_ratio(
            np.array([[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE])
        )
        
        # Determine if eyes are open
        self.left_eye_open = self.left_ear > self.blink_threshold
        self.right_eye_open = self.right_ear > self.blink_threshold
        
        # Pupil offsets (normalized)
        left_offset = (left_iris - left_eye_center) / w
        right_offset = (right_iris - right_eye_center) / w
        avg_offset = (left_offset + right_offset) / 2.0
        
        # Head pose
        nose = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])
        forehead = np.array([landmarks[10].x, landmarks[10].y, landmarks[10].z])
        
        # Face width
        left_face = np.array([landmarks[234].x, landmarks[234].y])
        right_face = np.array([landmarks[454].x, landmarks[454].y])
        face_width = np.linalg.norm(right_face - left_face)
        
        # Build 21 features matching training
        features = np.array([
            left_offset[0], left_offset[1],
            right_offset[0], right_offset[1],
            self.left_ear, self.right_ear,
            landmarks[LEFT_IRIS].x, landmarks[LEFT_IRIS].y,
            landmarks[RIGHT_IRIS].x, landmarks[RIGHT_IRIS].y,
            left_eye_center[0]/w, left_eye_center[1]/h,
            right_eye_center[0]/w, right_eye_center[1]/h,
            nose[0], nose[1], nose[2],
            forehead[0] - chin[0],
            left_eye_center[0]/w - right_eye_center[0]/w,
            face_width,
            self.left_ear - self.right_ear
        ], dtype=np.float32)
        
        return features, left_iris, right_iris, avg_offset, left_eye_points, right_eye_points
    
    def estimate_gaze_position(self, avg_offset):
        """Improved gaze estimation with better mapping"""
        # Increased sensitivity for better response
        scale_x = 20.0  # Much higher sensitivity
        scale_y = 15.0
        
        # Add exponential scaling for edges
        offset_x = avg_offset[0]
        offset_y = avg_offset[1]
        
        # Apply non-linear mapping for better edge detection
        if abs(offset_x) > 0.02:
            offset_x = offset_x * (1.0 + abs(offset_x) * 5.0)
        if abs(offset_y) > 0.02:
            offset_y = offset_y * (1.0 + abs(offset_y) * 5.0)
        
        gaze_x = self.screen_width / 2 + (offset_x * self.screen_width * scale_x)
        gaze_y = self.screen_height / 2 + (offset_y * self.screen_height * scale_y)
        
        # Clamp
        gaze_x = max(0, min(gaze_x, self.screen_width - 1))
        gaze_y = max(0, min(gaze_y, self.screen_height - 1))
        
        return np.array([gaze_x, gaze_y])
    
    def smooth_gaze(self, new_point):
        """Smooth with exponential moving average"""
        self.gaze_history.append(new_point)
        
        if len(self.gaze_history) == 0:
            return new_point
        
        # Weighted average - more weight to recent
        weights = np.exp(np.linspace(-2, 0, len(self.gaze_history)))
        weights /= weights.sum()
        
        smoothed = np.average(list(self.gaze_history), axis=0, weights=weights)
        return smoothed
    
    def predict_attention(self):
        """Predict with eyes-closed override"""
        # Check for eyes closed first
        both_eyes_closed = (not self.left_eye_open) and (not self.right_eye_open)
        if both_eyes_closed:
            self.eyes_closed_frames += 1
        else:
            self.eyes_closed_frames = 0
        
        # If eyes closed for >10 frames, force SLEEPING
        if self.eyes_closed_frames > 10:
            return 2, "SLEEPING", 0.95
        
        # Otherwise use model
        if self.attention_model is None or len(self.feature_buffer) < SEQUENCE_LENGTH:
            return 0, "FOCUSED", 0.5
        
        try:
            sequence = np.array(list(self.feature_buffer), dtype=np.float32)
            sequence = sequence.reshape(1, SEQUENCE_LENGTH, -1)
            
            prediction = self.attention_model.predict(sequence, verbose=0)
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
            status_map = {0: "FOCUSED", 1: "DISTRACTED", 2: "SLEEPING"}
            status = status_map[class_idx]
            
            if self.debug_mode and self.frame_count % 30 == 0:
                print(f"[ATTENTION] F:{prediction[0][0]:.2f} D:{prediction[0][1]:.2f} S:{prediction[0][2]:.2f}")
            
            return class_idx, status, confidence
            
        except Exception as e:
            if self.debug_mode and self.frame_count % 30 == 0:
                print(f"[ERROR] {e}")
            return 0, "FOCUSED", 0.5
    
    def draw_eye_landmarks(self, frame, left_eye_points, right_eye_points, left_iris, right_iris):
        """Draw detailed eye tracking visualization"""
        # Draw eye contours
        if self.show_landmarks:
            # Left eye
            for i in range(len(left_eye_points)):
                pt1 = tuple(left_eye_points[i].astype(int))
                pt2 = tuple(left_eye_points[(i+1) % len(left_eye_points)].astype(int))
                color = (0, 255, 0) if self.left_eye_open else (0, 0, 255)
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Right eye
            for i in range(len(right_eye_points)):
                pt1 = tuple(right_eye_points[i].astype(int))
                pt2 = tuple(right_eye_points[(i+1) % len(right_eye_points)].astype(int))
                color = (0, 255, 0) if self.right_eye_open else (0, 0, 255)
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw iris centers (large)
            cv2.circle(frame, tuple(left_iris.astype(int)), 5, (255, 255, 0), -1)
            cv2.circle(frame, tuple(right_iris.astype(int)), 5, (255, 255, 0), -1)
            
            # EAR text
            cv2.putText(frame, f"L:{self.left_ear:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"R:{self.right_ear:.2f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_ui(self, frame):
        """Draw enhanced UI"""
        # Canvas
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        canvas[:, :] = (10, 10, 10)
        
        # Webcam feed (larger for better visibility)
        webcam_w, webcam_h = 320, 240
        frame_large = cv2.resize(frame, (webcam_w, webcam_h))
        margin = 15
        x_pos = self.screen_width - webcam_w - margin
        y_pos = margin
        canvas[y_pos:y_pos+webcam_h, x_pos:x_pos+webcam_w] = frame_large
        cv2.rectangle(canvas, (x_pos-3, y_pos-3), 
                     (x_pos+webcam_w+3, y_pos+webcam_h+3), (100, 100, 100), 3)
        
        # Eye status indicators
        left_color = (0, 255, 0) if self.left_eye_open else (0, 0, 255)
        right_color = (0, 255, 0) if self.right_eye_open else (0, 0, 255)
        
        cv2.putText(canvas, "LEFT EYE", (x_pos, y_pos + webcam_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        cv2.putText(canvas, "RIGHT EYE", (x_pos + 150, y_pos + webcam_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        
        cv2.putText(canvas, "OPEN" if self.left_eye_open else "CLOSED", 
                   (x_pos, y_pos + webcam_h + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
        cv2.putText(canvas, "OPEN" if self.right_eye_open else "CLOSED", 
                   (x_pos + 150, y_pos + webcam_h + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        
        # Grid
        grid_color = (25, 25, 25)
        for i in range(1, 10):
            x = int(self.screen_width * i / 10)
            y = int(self.screen_height * i / 10)
            cv2.line(canvas, (x, 0), (x, self.screen_height), grid_color, 1)
            cv2.line(canvas, (0, y), (self.screen_width, y), grid_color, 1)
        
        # Center crosshair
        cx, cy = self.screen_width // 2, self.screen_height // 2
        cv2.line(canvas, (cx-40, cy), (cx+40, cy), (60, 60, 60), 3)
        cv2.line(canvas, (cx, cy-40), (cx, cy+40), (60, 60, 60), 3)
        cv2.circle(canvas, (cx, cy), 8, (60, 60, 60), -1)
        
        # GAZE DOT - HUGE AND VISIBLE
        gaze_x = int(self.gaze_point[0])
        gaze_y = int(self.gaze_point[1])
        
        # Multi-layer glow
        for radius, color in [
            (70, (0, 0, 20)),
            (55, (0, 0, 40)),
            (42, (0, 0, 80)),
            (32, (0, 0, 150)),
            (24, (0, 0, 230)),
            (18, (0, 0, 255)),
            (12, (50, 50, 255)),
            (7, (150, 150, 255)),
            (3, (255, 255, 255))
        ]:
            cv2.circle(canvas, (gaze_x, gaze_y), radius, color, -1)
        
        # Trail
        if len(self.gaze_history) > 1:
            points = list(self.gaze_history)
            for i in range(len(points) - 1):
                alpha = (i + 1) / len(points)
                thickness = max(2, int(10 * alpha))
                color = (0, int(50 * alpha), int(200 * alpha))
                pt1 = tuple(points[i].astype(int))
                pt2 = tuple(points[i+1].astype(int))
                cv2.line(canvas, pt1, pt2, color, thickness)
        
        # ATTENTION PANEL
        panel_x, panel_y = 20, self.screen_height - 250
        panel_w, panel_h = 500, 230
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)
        cv2.rectangle(canvas, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 3)
        
        y_off = panel_y + 40
        cv2.putText(canvas, "ATTENTION STATUS", (panel_x + 20, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        y_off += 50
        
        color_map = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 0, 255)}
        status_color = color_map[self.attention_class]
        
        cv2.putText(canvas, self.attention_status, (panel_x + 20, y_off),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 4)
        y_off += 50
        
        # Confidence bar
        bar_w = 440
        bar_h = 30
        bar_x = panel_x + 30
        cv2.rectangle(canvas, (bar_x, y_off), 
                     (bar_x + bar_w, y_off + bar_h), (50, 50, 50), 3)
        fill = int(bar_w * self.attention_confidence)
        if fill > 0:
            cv2.rectangle(canvas, (bar_x+2, y_off+2),
                         (bar_x + fill-2, y_off + bar_h-2), status_color, -1)
        y_off += 45
        
        cv2.putText(canvas, f"Confidence: {self.attention_confidence*100:.1f}%",
                   (bar_x, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        y_off += 35
        
        # Stats
        cv2.putText(canvas, f"FPS: {self.fps:.1f} | Buffer: {len(self.feature_buffer)}/{SEQUENCE_LENGTH}",
                   (bar_x, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        # Debug info
        if self.debug_mode:
            debug_text = [
                f"Gaze: ({gaze_x}, {gaze_y})",
                f"Left EAR: {self.left_ear:.3f}",
                f"Right EAR: {self.right_ear:.3f}",
                f"Eyes closed frames: {self.eyes_closed_frames}"
            ]
            for i, text in enumerate(debug_text):
                cv2.putText(canvas, text, (20, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return canvas
    
    def run(self):
        """Main loop"""
        try:
            print("\n🚀 Starting tracking...")
            print("👁️ Watch eyelid detection on webcam feed!")
            print("🔴 Red dot should follow your gaze\n")
            
            cv2.namedWindow('Eye Tracker', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Eye Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    features, left_iris, right_iris, avg_offset, left_eye_pts, right_eye_pts = \
                        self.extract_comprehensive_features(landmarks, frame.shape)
                    
                    # Draw landmarks on frame
                    self.draw_eye_landmarks(frame, left_eye_pts, right_eye_pts, left_iris, right_iris)
                    
                    self.feature_buffer.append(features)
                    
                    # Predict attention
                    class_idx, status, confidence = self.predict_attention()
                    self.attention_class = class_idx
                    self.attention_status = status
                    self.attention_confidence = confidence
                    
                    # Gaze estimation
                    new_gaze = self.estimate_gaze_position(avg_offset)
                    self.gaze_point = self.smooth_gaze(new_gaze)
                    
                    if self.debug_mode and self.frame_count % 30 == 0:
                        print(f"[GAZE] Offset: ({avg_offset[0]:.4f}, {avg_offset[1]:.4f}) → Screen: ({int(self.gaze_point[0])}, {int(self.gaze_point[1])})")
                
                else:
                    self.attention_status = "No Face"
                    self.attention_class = 1
                
                canvas = self.draw_ui(frame)
                
                current_time = time.time()
                if current_time - self.last_time > 0:
                    self.fps = 1.0 / (current_time - self.last_time)
                self.last_time = current_time
                self.frame_count += 1
                
                cv2.imshow('Eye Tracker', canvas)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self.feature_buffer.clear()
                    self.gaze_history.clear()
                    self.eyes_closed_frames = 0
                    print("[RESET]")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"[DEBUG] {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"[LANDMARKS] {'ON' if self.show_landmarks else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Done")


if __name__ == "__main__":
    try:
        tracker = ImprovedEyeTracker()
        tracker.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()