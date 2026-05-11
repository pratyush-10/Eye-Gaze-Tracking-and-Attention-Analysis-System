import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH     = 32
NUM_FEATURES        = 21
SCREEN_W, SCREEN_H  = 1920, 1080
CAM_PREVIEW_W       = 480
CAM_PREVIEW_H       = 360
GAZE_SENSITIVITY_X  = 15.0
GAZE_SENSITIVITY_Y  = 20.0

# How often to run each model (in frames).
# Attention/Gaze run every frame — they are fast and drive the heatmap.
# Emotion runs every 3 frames  → ~10 fps feel at 30 fps camera.
# Cognitive load runs every 6 frames  → heavier feature set.
EMOTION_INTERVAL    = 3
COGLOAD_INTERVAL    = 6

# Gaze heatmap blob
HEATMAP_RADIUS      = 220     # Radius (px) of the Gaussian blob
HEATMAP_ALPHA       = 0.70    # Blend strength (0=invisible, 1=opaque)

# Prediction smoothing window
SMOOTH_WINDOW = 8

# ==========================================
# UPDATED: FER-2013 Image Emotion Labels (7 Classes)
# ==========================================
EMOTION_LABELS  = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 
                   3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
COGLOAD_LABELS  = {0: 'Low', 1: 'Medium', 2: 'High'}


# ==========================================
# FEATURE EXTRACTORS  (unchanged logic)
# ==========================================
class LiveFeatureExtractor:
    @staticmethod
    def extract(gaze_history, head_history):
        gaze_arr = np.array(gaze_history)
        head_arr = np.array(head_history)
        seq_len  = len(gaze_arr)
        features_seq = []
        for t in range(seq_len):
            feats = list(gaze_arr[t]) + list(head_arr[t])
            feats += list(gaze_arr[t] - gaze_arr[t-1]) if t > 0 else [0., 0.]
            feats += list(head_arr[t] - head_arr[t-1]) if t > 0 else [0.]*6
            if t > 1:
                feats += list((gaze_arr[t]-gaze_arr[t-1]) - (gaze_arr[t-1]-gaze_arr[t-2]))
            else:
                feats += [0., 0.]
            if t > 0:
                feats.append(1. if np.linalg.norm(gaze_arr[t]-gaze_arr[t-1]) < 0.02 else 0.)
            else:
                feats.append(0.)
            feats.append(np.linalg.norm(head_arr[t][:3]))
            if t > 0:
                s = max(0, t-5)
                diffs = gaze_arr[s+1:t+1] - gaze_arr[s:t]
                feats.append(float(np.mean(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.)
            else:
                feats.append(0.)
            feats = feats[:NUM_FEATURES] + [0.]*(max(0, NUM_FEATURES-len(feats)))
            features_seq.append(feats[:NUM_FEATURES])
        return np.array(features_seq, dtype=np.float32)


class CognitiveLoadFeatureExtractor:
    @staticmethod
    def extract(gaze_history, head_history, lid_dist_history):
        gaze_arr = np.array(gaze_history)
        head_arr = np.array(head_history)
        lid_arr  = np.array(lid_dist_history)
        features = []

        for dim in range(2):
            col = gaze_arr[:, dim]
            features += [np.mean(col), np.std(col), np.var(col),
                         np.min(col),  np.max(col),
                         np.percentile(col,10), np.percentile(col,25),
                         np.percentile(col,50), np.percentile(col,75),
                         np.percentile(col,90), np.ptp(col),
                         np.median(col), np.mean(np.abs(col - np.mean(col)))]

        gaze_vel = (np.linalg.norm(np.diff(gaze_arr, axis=0), axis=1)
                    if len(gaze_arr) > 1 else np.array([0.]))
        for m in [np.mean, np.std, np.max, np.min, np.median]:
            features.append(float(m(gaze_vel)))
        features += list(np.percentile(gaze_vel, [10,25,50,75,90]))

        for vel in [np.abs(np.diff(gaze_arr[:,0])) if len(gaze_arr)>1 else np.array([0.]),
                    np.abs(np.diff(gaze_arr[:,1])) if len(gaze_arr)>1 else np.array([0.])]:
            features += [np.mean(vel), np.std(vel), np.max(vel), np.min(vel)]

        if len(gaze_vel) > 1:
            ga = np.abs(np.diff(gaze_vel))
            features += [np.mean(ga), np.std(ga), np.max(ga), np.min(ga)]
        else:
            features += [0.]*4

        fix = gaze_vel < 0.02; n = max(len(gaze_vel),1)
        features += [float(np.sum(fix)), float(len(gaze_vel)-np.sum(fix)),
                     float(np.sum(fix))/n, float(len(gaze_vel)-np.sum(fix))/n]
        features += [np.max(gaze_arr[:,0])-np.min(gaze_arr[:,0]),
                     np.max(gaze_arr[:,1])-np.min(gaze_arr[:,1]),
                     np.std(gaze_arr[:,0]), np.std(gaze_arr[:,1]),
                     np.percentile(gaze_vel,95), np.percentile(gaze_vel,5)]

        if len(gaze_vel) > 1:
            gj = np.abs(np.diff(gaze_vel))
            features += [np.mean(gj), np.std(gj), np.max(gj),
                         1./(1.+np.mean(gj))]
        else:
            features += [0., 0., 0., 1.]

        gpl = float(np.sum(np.linalg.norm(np.diff(gaze_arr, axis=0), axis=1)))
        features += [gpl, gpl/max(len(gaze_arr),1),
                     float(np.sum(gaze_vel**2)),
                     float(np.max(gaze_vel)-np.min(gaze_vel))]

        for dim in range(3):
            col = head_arr[:, dim]
            features += [np.mean(col), np.std(col), np.var(col),
                         np.min(col),  np.max(col),
                         np.percentile(col,25), np.percentile(col,50),
                         np.percentile(col,75)]
        for dim in [3, 4]:
            col = head_arr[:, dim]
            features += [np.mean(col), np.std(col), np.max(np.abs(col)),
                         np.percentile(col,25), np.percentile(col,75)]

        head_vel = (np.linalg.norm(np.diff(head_arr[:,:3], axis=0), axis=1)
                    if len(head_arr) > 1 else np.array([0.]))
        features += [np.mean(head_vel), np.std(head_vel), np.max(head_vel),
                     np.min(head_vel), np.percentile(head_vel,50),
                     np.percentile(head_vel,90)]

        if len(head_vel) > 1:
            ha = np.abs(np.diff(head_vel))
            features += [np.mean(ha), np.std(ha), np.max(ha)]
        else:
            features += [0.]*3

        hm = np.linalg.norm(head_arr[:,:3], axis=1)
        features += [np.mean(hm), np.std(hm), np.max(hm), np.min(hm)]

        if len(head_vel) > 1:
            hj = np.abs(np.diff(head_vel))
            features += [np.mean(hj), np.std(hj), np.max(hj)]
        else:
            features += [0.]*3

        hpl = float(np.sum(head_vel))
        features += [hpl, hpl/max(len(head_arr),1)]

        for p in [5,10,25,50,75,90,95]:
            features.append(float(np.percentile(lid_arr, p)))
        features += [np.mean(lid_arr), np.std(lid_arr), np.var(lid_arr),
                     np.min(lid_arr),  np.max(lid_arr),
                     np.ptp(lid_arr),  np.median(lid_arr)]

        blinks = lid_arr < 0.012
        bc = int(np.sum(blinks))
        features += [float(bc), bc/max(len(lid_arr),1),
                     float(np.sum(np.abs(np.diff(blinks.astype(int)))))]

        bi = np.where(blinks)[0]
        if len(bi) > 1:
            biv = np.diff(bi)
            features += [np.mean(biv), np.std(biv), np.max(biv), np.min(biv)]
        else:
            features += [0.]*4

        lv = np.abs(np.diff(lid_arr)) if len(lid_arr)>1 else np.array([0.])
        features += [np.mean(lv), np.std(lv), np.max(lv),
                     np.percentile(lv,50), np.percentile(lv,90)]
        features += [np.std(lid_arr), np.var(lid_arr),
                     float(np.sum(np.abs(np.diff(lid_arr)))) if len(lid_arr)>1 else 0.,
                     float(np.max(np.abs(np.diff(lid_arr)))) if len(lid_arr)>1 else 0.]
        features.append(float(np.sum(blinks > 0))/max(len(blinks),1))

        try:
            cx = float(np.corrcoef(gaze_arr[:,0], head_arr[:,0])[0,1])
            cy = float(np.corrcoef(gaze_arr[:,1], head_arr[:,1])[0,1])
            cx = 0. if np.isnan(cx) else cx
            cy = 0. if np.isnan(cy) else cy
        except Exception:
            cx, cy = 0., 0.
        features += [cx, cy]

        ws = max(1, len(gaze_arr)//4)
        for i in range(4):
            s, e = i*ws, min((i+1)*ws, len(gaze_arr))
            wd = gaze_arr[s:e] if s < e else gaze_arr[:1]
            features += [float(np.var(wd[:,0])), float(np.var(wd[:,1])),
                         float(np.mean(np.linalg.norm(np.diff(wd,axis=0),axis=1)))
                         if len(wd)>1 else 0.]

        om = np.linalg.norm(np.diff(gaze_arr, axis=0), axis=1)
        features += [np.mean(om), np.std(om), np.max(om), np.min(om),
                     np.percentile(om, 90)]
        features += [float(np.std(np.diff(gaze_vel))) if len(gaze_vel)>1 else 0.,
                     float(np.std(np.diff(head_vel))) if len(head_vel)>1 else 0.,
                     float(np.std(np.diff(lid_arr)))  if len(lid_arr)>1  else 0.,
                     float(np.sum(np.diff(gaze_arr,axis=0)**2)),
                     float(np.sum(np.diff(head_arr[:,:3],axis=0)**2))]
        features += [float(len(gaze_arr)),
                     float(np.mean([len(gaze_arr), len(head_arr), len(lid_arr)])),
                     gpl / (len(gaze_arr)+1e-7),
                     hpl / (len(head_arr)+1e-7)]

        if len(gaze_vel) > 0:
            gvn = (gaze_vel - gaze_vel.min()) / (gaze_vel.max()-gaze_vel.min()+1e-7)
            features += [float(np.sum(gvn*np.log(gvn+1e-7))/len(gaze_vel)),
                         float(np.sum(gvn**2)/len(gaze_vel))]
        else:
            features += [0., 0.]

        features = features[:414]
        features += [0.]*(414 - len(features))
        arr = np.array(features, dtype=np.float32)

        # z-score normalise + clip to avoid "always High" bias
        std = arr.std()
        if std > 1e-6:
            arr = (arr - arr.mean()) / std
        arr = np.clip(arr, -5., 5.)
        return arr


# ==========================================
# LIVE GAUSSIAN GAZE HEATMAP
# ==========================================
class GazeHeatmap:
    """
    Draws a single Gaussian probability blob centred on the current gaze point.
    Colour mapping:  centre → red (most likely)  →  yellow → green → blue  → transparent edge.
    No history accumulation — redrawn fresh every frame, fully real-time.
    """

    # Pre-bake a radial colormap:  index 0 = hot-centre (red), index 255 = cold-edge (blue)
    # We'll use OpenCV's JET reversed: JET goes blue→red, reverse gives red→blue.
    _CMAP = cv2.applyColorMap(
        np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET
    ).reshape(256, 3)   # shape (256, 3), BGR

    def __init__(self, screen_w: int, screen_h: int,
                 radius: int = HEATMAP_RADIUS, alpha: float = HEATMAP_ALPHA):
        self.W = screen_w
        self.H = screen_h
        self.radius = radius
        self.alpha  = alpha

        # Pre-compute a square distance kernel once.
        # kernel[y, x] = Gaussian weight in [0, 1], peak at centre.
        d = radius * 2 + 1
        cx = cy = radius
        yy, xx = np.mgrid[0:d, 0:d]
        dist2 = ((xx - cx)**2 + (yy - cy)**2).astype(np.float32)
        sigma  = radius / 2.5          # controls how tight / spread the blob is
        self._kernel = np.exp(-dist2 / (2 * sigma**2))   # range [0, 1]

    def render(self, canvas: np.ndarray, gaze_x: int, gaze_y: int) -> np.ndarray:
        """
        Blend a single Gaussian heatmap blob onto *canvas* at (gaze_x, gaze_y).
        Returns the canvas (modified in-place).
        """
        r = self.radius
        d = r * 2 + 1

        # --- compute visible rectangle on the canvas ---
        # Source (kernel) region
        k_x0 = max(0,  r - gaze_x)
        k_y0 = max(0,  r - gaze_y)
        k_x1 = d - max(0, (gaze_x + r + 1) - self.W)
        k_y1 = d - max(0, (gaze_y + r + 1) - self.H)

        # Destination (canvas) region
        c_x0 = max(0, gaze_x - r)
        c_y0 = max(0, gaze_y - r)
        c_x1 = min(self.W, gaze_x + r + 1)
        c_y1 = min(self.H, gaze_y + r + 1)

        if c_x1 <= c_x0 or c_y1 <= c_y0:
            return canvas

        kernel_crop = self._kernel[k_y0:k_y1, k_x0:k_x1]   # float32 [0,1]

        # Map kernel weight → colour index (0=red centre, 255=blue edge)
        idx = (kernel_crop * 255).astype(np.uint8)           # 255 at centre

        # Look up colour for every pixel in the crop
        colored = self._CMAP[idx]                            # shape (h, w, 3), BGR

        # Per-pixel alpha = kernel weight * global alpha
        alpha_map = (kernel_crop * self.alpha)[..., np.newaxis]   # (h, w, 1)

        # Blend
        dst = canvas[c_y0:c_y1, c_x0:c_x1].astype(np.float32)
        blended = dst * (1. - alpha_map) + colored.astype(np.float32) * alpha_map
        canvas[c_y0:c_y1, c_x0:c_x1] = blended.clip(0, 255).astype(np.uint8)
        return canvas


# ==========================================
# PROBABILITY SMOOTHER
# ==========================================
class ProbabilitySmoother:
    def __init__(self, num_classes: int, window: int = SMOOTH_WINDOW):
        self.history = deque(maxlen=window)

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.history.append(np.asarray(probs, dtype=np.float32))
        return np.mean(self.history, axis=0)


# ==========================================
# MAIN
# ==========================================
def main():
    print("Loading models …")
    try:
        model_attn    = tf.keras.models.load_model('models/attention_classifier.keras')
        model_gaze    = tf.keras.models.load_model('models/gaze_estimator.keras')
        
        # UPDATED: Changed model format to the newly trained .keras image model
        model_emotion = tf.keras.models.load_model('models/emotion_classifier.keras')
        model_cogload = tf.keras.models.load_model('models/cognitive_load_classifier.h5')
        print("All 4 models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # ---- Wrap inference calls with tf.function for speed ----
    @tf.function(reduce_retracing=True)
    def infer_emotion(x):
        return model_emotion(x, training=False)

    @tf.function(reduce_retracing=True)
    def infer_cogload(x):
        return model_cogload(x, training=False)

    @tf.function(reduce_retracing=True)
    def infer_attn(x):
        return model_attn(x, training=False)

    @tf.function(reduce_retracing=True)
    def infer_gaze(x):
        return model_gaze(x, training=False)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    window_name = 'Multimodal AI Tracker Dashboard'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)

    # ---- Histories ----
    gaze_history = deque(maxlen=SEQUENCE_LENGTH)
    head_history = deque(maxlen=SEQUENCE_LENGTH)
    lid_history  = deque(maxlen=SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        gaze_history.append([0.5, 0.5])
        head_history.append([0.0] * 6)
        lid_history.append(0.05)

    # ---- Helpers ----
    heatmap     = GazeHeatmap(SCREEN_W, SCREEN_H)
    
    # UPDATED: Change emotion smoother to accept 7 output classes (FER-2013)
    emo_smoother = ProbabilitySmoother(num_classes=7)
    cog_smoother = ProbabilitySmoother(num_classes=3)

    # Cached predictions (updated at their own cadence)
    emo_status  = "Detecting…"
    cog_status  = "Detecting…"
    cog_smoothed = np.ones(3, dtype=np.float32) / 3.

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame      = cv2.flip(frame, 1)
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # UPDATED: Generate grayscale frame for the emotion CNN model
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        results   = face_mesh.process(rgb_frame)
        h, w, _   = frame.shape

        current_gaze_raw = [0.5, 0.5]
        current_head_pose = [0.0] * 6
        eyes_closed = False
        lid_dist_avg = 0.0
        face_input = None # Placeholder for our cropped face tensor

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_lid  = abs(lm[159].y - lm[145].y)
            right_lid = abs(lm[386].y - lm[374].y)
            lid_dist_avg = (left_lid + right_lid) / 2.
            eyes_closed  = lid_dist_avg < 0.012

            nose = lm[1]
            yaw  = (nose.x - 0.5) * 2
            pitch = (nose.y - 0.5) * 2
            current_head_pose = [nose.x, nose.y, nose.z, pitch, yaw, 0.0]

            gaze_x = (lm[468].x + lm[473].x) / 2.
            gaze_y = (lm[468].y + lm[473].y) / 2.
            current_gaze_raw = [gaze_x, gaze_y]

            # ==========================================
            # UPDATED: Face Cropping for FER-2013 CNN
            # ==========================================
            # Find the min/max coordinates of the face landmarks to draw a bounding box
            x_coords = [int(l.x * w) for l in lm]
            y_coords = [int(l.y * h) for l in lm]
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

            # Add a slight pad around the face to make sure we don't cut off chin/forehead
            pad_x = int((x_max - x_min) * 0.1)
            pad_y = int((y_max - y_min) * 0.15)
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)

            # Crop from the grayscale frame
            face_crop = gray_frame[y_min:y_max, x_min:x_max]
            if face_crop.size > 0:
                # Resize to 48x48 (what FER expects), normalize 0-1, and format shape
                face_resized = cv2.resize(face_crop, (48, 48))
                face_arr = face_resized.astype(np.float32) / 255.0
                face_arr = np.expand_dims(face_arr, axis=-1)   # Channels dim: (48, 48, 1)
                face_input = np.expand_dims(face_arr, axis=0)  # Batch dim: (1, 48, 48, 1)

            # Render Lid indicators
            line_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            cv2.line(frame,
                     (int(lm[159].x*w), int(lm[159].y*h)),
                     (int(lm[145].x*w), int(lm[145].y*h)), line_color, 2)
            cv2.line(frame,
                     (int(lm[386].x*w), int(lm[386].y*h)),
                     (int(lm[374].x*w), int(lm[374].y*h)), line_color, 2)

        # ---- Update histories ----
        gaze_history.append(current_gaze_raw)
        head_history.append(current_head_pose)
        lid_history.append(lid_dist_avg)

        # ---- Model 1 & 2: Attention + Gaze (every frame, fast) ----
        feat_seq = LiveFeatureExtractor.extract(gaze_history, head_history)
        feat_tensor = tf.constant(feat_seq[np.newaxis], dtype=tf.float32)
        attn_probs = infer_attn(feat_tensor).numpy()[0]

        # ==========================================
        # UPDATED: Model 3: Emotion (Using CNN on Image)
        # ==========================================
        if frame_idx % EMOTION_INTERVAL == 0:
            if face_input is not None:
                emo_tensor = tf.constant(face_input, dtype=tf.float32)
                emo_raw    = infer_emotion(emo_tensor).numpy()[0]
                emo_smoothed_probs = emo_smoother.update(emo_raw)
                emo_status = EMOTION_LABELS.get(int(np.argmax(emo_smoothed_probs)), "Unknown")
            else:
                emo_status = "No Face Found"

        # ---- Model 4: Cognitive Load (every COGLOAD_INTERVAL frames) ----
        if frame_idx % COGLOAD_INTERVAL == 0:
            cog_feat   = CognitiveLoadFeatureExtractor.extract(gaze_history, head_history, lid_history)
            cog_tensor = tf.constant(cog_feat[np.newaxis], dtype=tf.float32)
            cog_raw    = infer_cogload(cog_tensor).numpy()[0]
            cog_smoothed = cog_smoother.update(cog_raw)
            cog_status = COGLOAD_LABELS.get(int(np.argmax(cog_smoothed)), "Unknown")

        frame_idx += 1

        # ---- Gaze → screen coords ----
        dx = current_gaze_raw[0] - 0.5
        dy = current_gaze_raw[1] - 0.5
        screen_x = int(SCREEN_W/2 + dx * SCREEN_W  * GAZE_SENSITIVITY_X)
        screen_y = int(SCREEN_H/2 + dy * SCREEN_H  * GAZE_SENSITIVITY_Y)
        looking_on_screen = (0 <= screen_x <= SCREEN_W) and (0 <= screen_y <= SCREEN_H)
        screen_x_c = max(0, min(SCREEN_W-1, screen_x))
        screen_y_c = max(0, min(SCREEN_H-1, screen_y))

        # ---- Attention status ----
        if eyes_closed:
            attn_status, attn_color = "SLEEPING",   (0,  0,  255)
        elif looking_on_screen:
            attn_status, attn_color = "ATTENTIVE",  (0, 255,   0)
        else:
            attn_status, attn_color = "DISTRACTED", (0, 165, 255)

        # ==========================================
        # RENDER
        # ==========================================
        main_canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        # 1. Gaze heatmap blob (fresh every frame, no accumulation)
        if not eyes_closed:
            heatmap.render(main_canvas, screen_x_c, screen_y_c)

        # 2. Camera PIP
        cam_preview = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))
        cv2.rectangle(cam_preview, (0, 0),
                      (CAM_PREVIEW_W-1, CAM_PREVIEW_H-1), attn_color, 4)
        cv2.putText(cam_preview, f"Lid: {lid_dist_avg:.3f}",
                    (10, CAM_PREVIEW_H-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        y0 = 20; x0 = SCREEN_W - CAM_PREVIEW_W - 20
        main_canvas[y0:y0+CAM_PREVIEW_H, x0:x0+CAM_PREVIEW_W] = cam_preview

        # 3. Dashboard
        cv2.putText(main_canvas, "REAL-TIME MULTIMODAL ANALYSIS",
                    (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2)
        cv2.line(main_canvas, (50, 90), (660, 90), (255,255,255), 2)

        cv2.putText(main_canvas, "ATTENTION :", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        cv2.putText(main_canvas, attn_status, (270, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, attn_color, 3)

        cv2.putText(main_canvas, "EMOTION   :", (50, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        cv2.putText(main_canvas, emo_status.upper(), (270, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

        cv2.putText(main_canvas, "COG. LOAD :", (50, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        cv2.putText(main_canvas, cog_status.upper(), (270, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,105,180), 3)

        # Cognitive load confidence bars
        bar_y = 285; bx = 50; bw = 300
        bar_cols = [(100,220,100), (50,180,255), (80,80,255)]
        for i, (label, prob) in enumerate(zip(COGLOAD_LABELS.values(), cog_smoothed)):
            cv2.rectangle(main_canvas,
                          (bx, bar_y + i*22), (bx + int(bw*prob), bar_y + i*22 + 14),
                          bar_cols[i], -1)
            cv2.putText(main_canvas, f"{label} {prob:.2f}",
                        (bx + bw + 8, bar_y + i*22 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        if attn_status == "SLEEPING":
            cv2.putText(main_canvas, "WAKE UP!",
                        (SCREEN_W//2 - 200, SCREEN_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 10)

        cv2.imshow(window_name, main_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()