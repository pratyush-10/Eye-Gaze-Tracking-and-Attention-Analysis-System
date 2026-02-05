"""
Real-time eye gaze and attention tracker with UI
Uses laptop camera to predict attention state in real-time
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import threading
from collections import deque
from config import *
from feature_engineering import FeatureExtractor


class RealtimeTracker:
    """Real-time gaze and attention tracker"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Eye Gaze & Attention Tracker")
        self.root.geometry("1200x800")
        self.root.config(bg='#2b2b2b')
        
        # Load models
        self.load_models()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Feature buffer (store last 32 frames)
        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
        
        # UI Setup
        self.setup_ui()
        
        # Start camera thread
        self.is_running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Schedule UI update
        self.update_ui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def load_models(self):
        """Load trained models"""
        print("Loading trained models...")
        try:
            self.model_attn = keras.models.load_model(
                str(MODELS_DIR / 'attention_classifier.keras')
            )
            self.model_gaze = keras.models.load_model(
                str(MODELS_DIR / 'gaze_estimator.keras')
            )
            print("✓ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model_attn = None
            self.model_gaze = None
    
    def setup_ui(self):
        """Setup tkinter UI"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame (left side)
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_label = ttk.Label(video_frame, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Stats frame (right side)
        stats_frame = ttk.LabelFrame(main_frame, text="Predictions", padding=15)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        
        # Attention State
        ttk.Label(stats_frame, text="Attention State:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.attn_label = ttk.Label(
            stats_frame, 
            text="Initializing...",
            font=('Arial', 14, 'bold'),
            foreground='blue'
        )
        self.attn_label.pack(anchor=tk.W, pady=5)
        
        # Confidence
        ttk.Label(stats_frame, text="Confidence:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.conf_label = ttk.Label(stats_frame, text="0%", font=('Arial', 12))
        self.conf_label.pack(anchor=tk.W, pady=5)
        
        # Confidence bar
        self.conf_bar = ttk.Progressbar(stats_frame, length=200, mode='determinate')
        self.conf_bar.pack(fill=tk.X, pady=5)
        
        # Gaze coordinates
        ttk.Label(stats_frame, text="Gaze Position:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.gaze_label = ttk.Label(stats_frame, text="X: 0.00, Y: 0.00", font=('Arial', 11))
        self.gaze_label.pack(anchor=tk.W, pady=5)
        
        # Gaze indicator (visual)
        self.gaze_canvas = tk.Canvas(stats_frame, width=200, height=150, bg='#1a1a1a')
        self.gaze_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        self.gaze_canvas.create_rectangle(0, 0, 200, 150, fill='#333333')
        self.gaze_point = self.gaze_canvas.create_oval(95, 70, 105, 80, fill='red')
        
        # Frame count
        ttk.Label(stats_frame, text="Buffer Status:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.buffer_label = ttk.Label(stats_frame, text="0/32 frames", font=('Arial', 11))
        self.buffer_label.pack(anchor=tk.W, pady=5)
        
        # Buffer bar
        self.buffer_bar = ttk.Progressbar(stats_frame, length=200, mode='determinate', maximum=SEQUENCE_LENGTH)
        self.buffer_bar.pack(fill=tk.X, pady=5)
        
        # Status
        ttk.Label(stats_frame, text="Status:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.status_label = ttk.Label(stats_frame, text="Initializing...", font=('Arial', 11), foreground='orange')
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # FPS
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0", font=('Arial', 10), foreground='gray')
        self.fps_label.pack(anchor=tk.W, pady=(20, 0))
    
    def extract_frame_features(self, frame):
        """Extract features from single frame"""
        try:
            # Resize to 224x224
            frame_resized = cv2.resize(frame, (224, 224))
            
            # Normalize to [0, 1]
            frame_norm = frame_resized.astype(np.float32) / 255.0
            
            # Flatten to 1D vector
            features = frame_norm.flatten()
            
            # Reduce to 21 features (compress from 224*224*3)
            if len(features) > 21:
                features = features[::len(features)//21][:21]
            else:
                features = np.pad(features, (0, 21 - len(features)))
            
            return features.astype(np.float32)
        except:
            return np.zeros(21, dtype=np.float32)
    
    def camera_loop(self):
        """Capture frames from camera in separate thread"""
        import time
        frame_times = deque(maxlen=30)
        
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                self.status_text = "Camera Error"
                continue
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Extract features from frame
            features = self.extract_frame_features(frame)
            self.feature_buffer.append(features)
            self.frame_count += 1
            
            # Store frame for display
            self.current_frame = frame.copy()
            
            # Make predictions when buffer is full
            if len(self.feature_buffer) == SEQUENCE_LENGTH:
                try:
                    # Prepare input
                    X_input = np.array([list(self.feature_buffer)]).astype(np.float32)
                    
                    # Get predictions
                    attn_pred = self.model_attn.predict(X_input, verbose=0)
                    gaze_pred = self.model_gaze.predict(X_input, verbose=0)
                    
                    # Process attention
                    attn_class = np.argmax(attn_pred[0])
                    attn_conf = float(np.max(attn_pred[0])) * 100
                    
                    attn_names = ['Focused', 'Distracted', 'Sleeping']
                    self.attn_text = attn_names[attn_class]
                    self.conf_text = f"{attn_conf:.1f}%"
                    self.conf_value = attn_conf
                    
                    # Process gaze
                    gaze_x, gaze_y = gaze_pred[0]
                    self.gaze_x = float(gaze_x)
                    self.gaze_y = float(gaze_y)
                    
                    self.status_text = f"Predicting ({len(self.feature_buffer)}/{SEQUENCE_LENGTH})"
                
                except Exception as e:
                    self.status_text = f"Prediction Error: {str(e)[:30]}"
            else:
                self.status_text = f"Buffering ({len(self.feature_buffer)}/{SEQUENCE_LENGTH})"
            
            # Calculate FPS
            frame_times.append(time.time())
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0] + 0.001)
                self.fps_text = f"{fps:.1f}"
            
            time.sleep(0.01)  # ~30 FPS
    
    def update_ui(self):
        """Update UI with latest predictions"""
        if hasattr(self, 'current_frame'):
            # Convert frame for display
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.video_label.config(image=photo)
            self.video_label.image = photo
        
        # Update attention label
        if hasattr(self, 'attn_text'):
            colors = {'Focused': 'green', 'Distracted': 'orange', 'Sleeping': 'red'}
            color = colors.get(self.attn_text, 'blue')
            self.attn_label.config(text=self.attn_text, foreground=color)
        
        # Update confidence
        if hasattr(self, 'conf_text'):
            self.conf_label.config(text=self.conf_text)
        if hasattr(self, 'conf_value'):
            self.conf_bar['value'] = self.conf_value
        
        # Update gaze
        if hasattr(self, 'gaze_x') and hasattr(self, 'gaze_y'):
            self.gaze_label.config(
                text=f"X: {self.gaze_x:.2f}, Y: {self.gaze_y:.2f}"
            )
            # Move point on canvas
            x = int(self.gaze_x * 200)
            y = int(self.gaze_y * 150)
            x = max(0, min(200, x))
            y = max(0, min(150, y))
            self.gaze_canvas.coords(self.gaze_point, x-5, y-5, x+5, y+5)
        
        # Update buffer status
        buffer_len = len(self.feature_buffer)
        self.buffer_label.config(text=f"{buffer_len}/{SEQUENCE_LENGTH} frames")
        self.buffer_bar['value'] = buffer_len
        
        # Update status
        if hasattr(self, 'status_text'):
            self.status_label.config(text=self.status_text)
        
        # Update FPS
        if hasattr(self, 'fps_text'):
            self.fps_label.config(text=f"FPS: {self.fps_text}")
        
        # Schedule next update
        self.root.after(50, self.update_ui)
    
    def on_close(self):
        """Handle window close"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# Run application
if __name__ == "__main__":
    root = tk.Tk()
    app = RealtimeTracker(root)
    root.mainloop()
