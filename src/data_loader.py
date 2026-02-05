"""
Data loader for MPIIGaze dataset
Reads image files and annotations from the MPIIGaze dataset
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import scipy.io as sio
import json
from config import *

class MPIIGazeDataLoader:
    """Load and manage MPIIGaze dataset"""
    
    def __init__(self, data_root=DATA_ROOT):
        self.data_root = Path(data_root)
        self.original_data_path = self.data_root / "Data" / "Original"
        self.calibration_path = self.data_root.parent / "Calibration"
    
    def load_annotation_file(self, annotation_file_path):
        """Load annotation.txt file and parse all lines"""
        annotations = []
        
        with open(annotation_file_path, 'r') as f:
            for line in f:
                ann = self._parse_annotation_line(line)
                annotations.append(ann)
        
        return annotations
    
    @staticmethod
    def _parse_annotation_line(line):
        """
        Parse a single line from annotation.txt
        Each line has 41 numbers
        """
        values = line.strip().split()
        values = np.array([float(v) for v in values])
        
        return {
            'eye_landmarks': values[0:24],      # 12 points × 2 coords
            'gaze_2d_screen': values[24:26],    # x, y on screen
            'gaze_3d_camera': values[26:29],    # x, y, z from camera
            'head_pose': values[29:35],         # head rotation + position
            'right_eye_3d': values[35:38],      # right eye position
            'left_eye_3d': values[38:41]        # left eye position
        }
    
    def load_calibration_data(self, participant_id):
        """Load calibration files for a participant"""
        calib_dir = self.calibration_path / participant_id
        
        calibration = {}
        
        # Load camera parameters
        camera_file = calib_dir / "Camera.mat"
        if camera_file.exists():
            camera_data = sio.loadmat(str(camera_file))
            calibration['camera_matrix'] = camera_data['cameraMatrix']
            calibration['dist_coeffs'] = camera_data['distCoeffs']
        
        # Load monitor pose
        monitor_file = calib_dir / "monitorPose.mat"
        if monitor_file.exists():
            monitor_data = sio.loadmat(str(monitor_file))
            calibration['monitor_rvecs'] = monitor_data['rvecs']
            calibration['monitor_tvecs'] = monitor_data['tvecs']
        
        # Load screen size
        screen_file = calib_dir / "screenSize.mat"
        if screen_file.exists():
            screen_data = sio.loadmat(str(screen_file))
            calibration['screen_height_pixel'] = float(
                screen_data['height_pixel']
            )
            calibration['screen_width_pixel'] = float(
                screen_data['width_pixel']
            )
            calibration['screen_height_mm'] = float(
                screen_data['height_mm']
            )
            calibration['screen_width_mm'] = float(
                screen_data['width_mm']
            )
        
        return calibration
    
    def collect_all_samples(self):
        """
        Traverse all participants and days
        Returns list of all image-annotation pairs
        """
        all_samples = []
        
        for participant in PARTICIPANTS:
            participant_dir = self.original_data_path / participant
            
            if not participant_dir.exists():
                print(f"⚠ Participant {participant} not found")
                continue
            
            # Load calibration once per participant
            calibration = self.load_calibration_data(participant)
            
            # Iterate through days
            days = sorted([d for d in os.listdir(participant_dir)
                          if os.path.isdir(participant_dir / d)])
            
            for day in days:
                day_dir = participant_dir / day
                annotation_file = day_dir / "annotation.txt"
                
                if not annotation_file.exists():
                    continue
                
                # Load all annotations for this day
                annotations = self.load_annotation_file(annotation_file)
                
                # Get image list
                images = sorted([f for f in os.listdir(day_dir)
                               if f.endswith('.jpg')])
                
                # Match images to annotations
                for img_idx, img_file in enumerate(images):
                    if img_idx < len(annotations):
                        image_path = day_dir / img_file
                        
                        sample = {
                            'participant_id': participant,
                            'day': day,
                            'image_index': img_idx,
                            'image_path': str(image_path),
                            'annotation': annotations[img_idx],
                            'calibration': calibration
                        }
                        
                        all_samples.append(sample)
        
        print(f"✓ Collected {len(all_samples)} samples")
        return all_samples
    
    def load_image(self, image_path):
        """Load and preprocess single image"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
