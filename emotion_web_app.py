import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.spatial import Delaunay
import threading
import time
import pickle
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string
import os

# Try to import XGBoost (needed if model was trained with XGBoost)
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    # Will be handled during model loading

# Import scikit-learn components (needed for model loading)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Warning: scikit-learn not found. Install with: pip install scikit-learn")

# ==================== MediaPipe Initialization ====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

# ==================== Landmark Indices ====================
left_brow_idx = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
right_brow_idx = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
inner_brow_idx = [63, 293]
brow_landmarks_idx = list(set(left_brow_idx + right_brow_idx + inner_brow_idx))

left_cheek_idx = [205, 206, 216, 204, 207, 114, 115, 116, 213, 214, 215]
right_cheek_idx = [425, 426, 436, 424, 427, 343, 344, 345, 433, 434, 435]
cheek_landmarks_idx = list(set(left_cheek_idx + right_cheek_idx))

left_eye_idx = [33, 160, 158, 133, 153, 144, 145, 159]
right_eye_idx = [362, 385, 387, 263, 373, 374, 380, 386]
outer_eye_idx = [33, 133, 362, 263]
eye_landmarks_idx = list(set(left_eye_idx + right_eye_idx + outer_eye_idx))

jaw_landmarks_idx = [152, 176, 136, 172, 397, 365, 366, 379, 400, 378, 377]

lip_landmarks_idx = [13, 14, 37, 39, 40, 61, 78, 80, 81, 82, 84, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 321, 324, 375, 402, 405, 409, 415]
mouth_landmarks_idx = [13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

# Surface splits
left_brow_idx_surface = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
right_brow_idx_surface = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
left_cheek_idx_surface = [205, 206, 216, 204, 207, 114, 115, 116]
right_cheek_idx_surface = [425, 426, 436, 424, 427, 343, 344, 345]
left_eye_idx_surface = [33, 160, 158, 133, 153, 144, 145, 159]
right_eye_idx_surface = [362, 385, 387, 263, 373, 374, 380, 386]
left_jaw_idx_surface = [152, 176, 136, 172]
right_jaw_idx_surface = [397, 365, 366, 379, 400, 378, 377]
left_lip_idx_surface = [61, 78, 80, 81, 82, 84, 91, 95]
right_lip_idx_surface = [291, 308, 310, 311, 312, 314, 321, 324]
left_mouth_idx_surface = [61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 17]
right_mouth_idx_surface = [291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375]

# ==================== Buffers ====================
buffer_size = 10
brow_raise_buffer = deque(maxlen=buffer_size)
brow_left_raise_buffer = deque(maxlen=buffer_size)
brow_right_raise_buffer = deque(maxlen=buffer_size)
brow_inner_raise_buffer = deque(maxlen=buffer_size)
brow_vel_buffer = deque(maxlen=buffer_size)
brow_surface_var_buffer = deque(maxlen=buffer_size)
brow_surface_dir_buffer = deque(maxlen=buffer_size)

cheek_raise_buffer = deque(maxlen=buffer_size)
cheek_vel_buffer = deque(maxlen=buffer_size)
cheek_surface_var_buffer = deque(maxlen=buffer_size)
cheek_surface_dir_buffer = deque(maxlen=buffer_size)

eye_ratio_buffer = deque(maxlen=buffer_size)
eye_vel_buffer = deque(maxlen=buffer_size)
blink_buffer = deque(maxlen=30)
eye_surface_var_buffer = deque(maxlen=buffer_size)
eye_surface_dir_buffer = deque(maxlen=buffer_size)

jaw_open_buffer = deque(maxlen=buffer_size)
jaw_vel_buffer = deque(maxlen=buffer_size)
jaw_surface_var_buffer = deque(maxlen=buffer_size)
jaw_surface_dir_buffer = deque(maxlen=buffer_size)

lips_open_buffer = deque(maxlen=buffer_size)
lips_vel_buffer = deque(maxlen=buffer_size)
lips_surface_var_buffer = deque(maxlen=buffer_size)
lips_surface_dir_buffer = deque(maxlen=buffer_size)

mouth_open_buffer = deque(maxlen=buffer_size)
mouth_vel_buffer = deque(maxlen=buffer_size)
mouth_surface_var_buffer = deque(maxlen=buffer_size)
mouth_surface_dir_buffer = deque(maxlen=buffer_size)

# ==================== Global Variables ====================
frame_global = None
features_global = {}
prev_landmarks_global = None
recording = False
recorded_features = []
trained_model = None
label_encoder = None
feature_scaler = None
recording_frame_count = 0  # Track frames since recording started

# ==================== Load Saved Model ====================
def load_model():
    global trained_model, label_encoder, feature_scaler
    try:
        model_path = "best_model.pkl"
        encoder_path = "label_encoder.pkl"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Warning: {model_path} not found. Please train the model first.")
            return False
        
        if not os.path.exists(encoder_path):
            print(f"⚠️  Warning: {encoder_path} not found. Please train the model first.")
            return False
        
        # Check for required dependencies
        missing_deps = []
        
        try:
            import xgboost
            print("✅ XGBoost available")
        except ImportError:
            print("⚠️  XGBoost not found (may be needed if model is XGBoost)")
            missing_deps.append("xgboost")
        
        try:
            import sklearn
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import LabelEncoder
            print("✅ scikit-learn available")
        except ImportError:
            print("❌ scikit-learn not found (REQUIRED)")
            missing_deps.append("scikit-learn")
            return False
        
        if missing_deps and 'xgboost' in missing_deps:
            print("⚠️  Warning: XGBoost not found. Model may fail to load if it's an XGBoost model.")
            print("   Install with: pip install xgboost")
        
        print("Loading model, encoder, and scaler...")
        with open(model_path, "rb") as f:
            trained_model = pickle.load(f)
        
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        
        # Load scaler if it exists
        scaler_path = "feature_scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                feature_scaler = pickle.load(f)
            print("✅ Model, encoder, and scaler loaded successfully!")
        else:
            print("⚠️  Warning: feature_scaler.pkl not found. Using unnormalized features (may be less accurate).")
            print("   Please retrain the model to generate the scaler.")
            feature_scaler = None
        
        print(f"   Model type: {type(trained_model).__name__}")
        return True
    except ModuleNotFoundError as e:
        error_msg = str(e).lower()
        if 'xgboost' in error_msg:
            print(f"\n❌ Error: XGBoost is required to load this model.")
            print("   The saved model was trained with XGBoost.")
            print("\n   To fix this, install XGBoost:")
            print("   pip install xgboost")
            print("\n   Or if using a virtual environment:")
            print("   (.venv) pip install xgboost")
            return False
        elif 'sklearn' in error_msg or 'scikit-learn' in error_msg:
            print(f"\n❌ Error: scikit-learn is required to load this model.")
            print("   The saved model was trained with scikit-learn.")
            print("\n   To fix this, install scikit-learn:")
            print("   pip install scikit-learn")
            print("\n   Or if using a virtual environment:")
            print("   (.venv) pip install scikit-learn")
            return False
        else:
            print(f"❌ Error loading model: Missing module - {str(e)}")
            print("\n   Common missing dependencies:")
            print("   - scikit-learn: pip install scikit-learn")
            print("   - xgboost: pip install xgboost")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("\n   Troubleshooting:")
        print("   1. Ensure best_model.pkl and label_encoder.pkl exist")
        print("   2. If model was trained with XGBoost, install it: pip install xgboost")
        print("   3. Try retraining the model if issues persist")
        return False

# ==================== Feature Computation Functions ====================
def compute_surface_vectors_split(landmarks, prev_landmarks, left_idx, right_idx):
    if prev_landmarks is None or landmarks is None:
        zero = {'vectors': np.array([]), 'positions': [], 'norms': np.array([]), 'mean_mag': 0.0, 'var': 0.0, 'angle': 0.0}
        return {'left': zero, 'right': zero}

    def process_side(idx_list):
        if not idx_list:
            return {'vectors': np.array([]), 'positions': [], 'norms': np.array([]), 'mean_mag': 0.0, 'var': 0.0, 'angle': 0.0}

        curr_pos = []
        prev_pos = []
        for idx in idx_list:
            if idx >= len(landmarks) or idx >= len(prev_landmarks):
                continue
            curr_pos.append(np.array(landmarks[idx]))
            prev_pos.append(np.array(prev_landmarks[idx]))

        if len(curr_pos) < 3:
            return {'vectors': np.array([]), 'positions': [], 'norms': np.array([]), 'mean_mag': 0.0, 'var': 0.0, 'angle': 0.0}

        points2d = np.array([p[:2] for p in curr_pos])

        try:
            tri = Delaunay(points2d)
        except:
            return {'vectors': np.array([]), 'positions': [], 'norms': np.array([]), 'mean_mag': 0.0, 'var': 0.0, 'angle': 0.0}

        triangle_norms = []
        triangle_vectors = []
        triangle_areas = []

        for simplex in tri.simplices:
            i1, i2, i3 = simplex
            v1 = curr_pos[i1] - prev_pos[i1]
            v2 = curr_pos[i2] - prev_pos[i2]
            v3 = curr_pos[i3] - prev_pos[i3]
            mean_v = (v1 + v2 + v3) / 3
            norm = np.linalg.norm(mean_v)
            triangle_norms.append(norm)
            if norm > 1e-6:
                triangle_vectors.append(mean_v / norm)
            else:
                triangle_vectors.append(mean_v)
            area = 0.5 * np.abs(np.cross(points2d[i2] - points2d[i1], points2d[i3] - points2d[i1]))
            triangle_areas.append(area)

        triangle_norms = np.array(triangle_norms)
        mean_mag = np.mean(triangle_norms) if len(triangle_norms) > 0 else 0.0
        var = np.var(triangle_norms) if len(triangle_norms) > 0 else 0.0

        angle = 0.0
        if triangle_vectors and triangle_areas:
            weighted_vectors = np.array(triangle_vectors) * np.array(triangle_areas)[:, np.newaxis]
            avg = np.sum(weighted_vectors[:, :2], axis=0) / np.sum(triangle_areas)
            n = np.linalg.norm(avg)
            if n > 1e-6:
                angle = np.arctan2(avg[1], avg[0])

        return {'vectors': np.array(triangle_vectors) if triangle_vectors else np.array([]),
                'positions': [np.mean([curr_pos[i] for i in s], axis=0) for s in tri.simplices],
                'norms': triangle_norms,
                'mean_mag': mean_mag, 'var': var, 'angle': angle}

    return {'left': process_side(left_idx), 'right': process_side(right_idx)}

def compute_brow_features(landmarks, prev_landmarks):
    if landmarks is None: return {}
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    left_brow_ys = [norm_landmarks[i][1] for i in left_brow_idx]
    right_brow_ys = [norm_landmarks[i][1] for i in right_brow_idx]
    inner_brow_ys = [norm_landmarks[i][1] for i in inner_brow_idx]

    left_raise = -np.mean(left_brow_ys)
    right_raise = -np.mean(right_brow_ys)
    inner_raise = -np.mean(inner_brow_ys)
    overall_raise = (left_raise + right_raise) / 2

    brow_left_raise_buffer.append(left_raise)
    brow_right_raise_buffer.append(right_raise)
    brow_inner_raise_buffer.append(inner_raise)
    brow_raise_buffer.append(overall_raise)

    left_stats = [np.mean(brow_left_raise_buffer), np.std(brow_left_raise_buffer)] if len(brow_left_raise_buffer) > 1 else [0, 0]
    right_stats = [np.mean(brow_right_raise_buffer), np.std(brow_right_raise_buffer)] if len(brow_right_raise_buffer) > 1 else [0, 0]
    inner_stats = [np.mean(brow_inner_raise_buffer), np.std(brow_inner_raise_buffer)] if len(brow_inner_raise_buffer) > 1 else [0, 0]

    brow_vel = abs(overall_raise - brow_raise_buffer[-2]) if len(brow_raise_buffer) > 1 else 0
    brow_vel_buffer.append(brow_vel)
    vel_stats = [np.mean(brow_vel_buffer), np.std(brow_vel_buffer)] if len(brow_vel_buffer) > 1 else [0, 0]
    rapid_count = len(find_peaks(list(brow_vel_buffer), distance=2)[0]) if len(brow_vel_buffer) > 1 else 0

    micro_var = np.var(brow_raise_buffer) if len(brow_raise_buffer) > 1 else 0.0
    freq_mean = np.mean(np.abs(fft(list(brow_raise_buffer)))[:buffer_size//2]) if len(brow_raise_buffer) == buffer_size else 0.0
    peak_freq = np.max(np.abs(fft(list(brow_raise_buffer)))[:buffer_size//2]) if len(brow_raise_buffer) == buffer_size else 0.0

    brow_asym = abs(left_raise - right_raise)
    inner_asym = abs(norm_landmarks[63][1] - norm_landmarks[293][1])
    asym_diffs = np.abs(np.array(brow_left_raise_buffer) - np.array(brow_right_raise_buffer))
    temp_asym_var = np.var(asym_diffs) if len(asym_diffs) > 1 else 0.0

    surface = compute_surface_vectors_split(landmarks, prev_landmarks, left_brow_idx_surface, right_brow_idx_surface)
    left, right = surface['left'], surface['right']

    brow_surface_var_buffer.append({'left': left['var'], 'right': right['var']})
    brow_surface_dir_buffer.append({'left': left['angle'], 'right': right['angle']})

    left_vars = [x['left'] for x in list(brow_surface_var_buffer)[-10:]]
    right_vars = [x['right'] for x in list(brow_surface_var_buffer)[-10:]]
    left_angles = [x['left'] for x in list(brow_surface_dir_buffer)[-10:]]
    right_angles = [x['right'] for x in list(brow_surface_dir_buffer)[-10:]]

    var_stats_l = [np.mean(left_vars), np.std(left_vars), np.min(left_vars), np.max(left_vars)] if left_vars else [0]*4
    var_stats_r = [np.mean(right_vars), np.std(right_vars), np.min(right_vars), np.max(right_vars)] if right_vars else [0]*4
    dir_stats_l = [np.mean(left_angles), np.std(left_angles)] if len(left_angles) > 1 else [0, 0]
    dir_stats_r = [np.mean(right_angles), np.std(right_angles)] if len(right_angles) > 1 else [0, 0]

    return {
        'Brow micro-expression variance mean': micro_var,
        'Brow micro-expression rapid changes count': rapid_count,
        'Brow velocity (mean)': vel_stats[0],
        'Brow velocity (std)': vel_stats[1],
        'Right brow raise (mean)': right_stats[0],
        'Right brow raise (std)': right_stats[1],
        'Left brow raise (mean)': left_stats[0],
        'Left brow raise (std)': left_stats[1],
        'Inner brow raise (mean)': inner_stats[0],
        'Inner brow raise (std)': inner_stats[1],
        'Brow asymmetry (mean)': brow_asym,
        'Temporal brow asymmetry variance': temp_asym_var,
        'Brow frequency mean': freq_mean,
        'Brow peak frequency': peak_freq,
        'Left surface vector magnitude mean': left['mean_mag'],
        'Left surface variance (current)': left['var'],
        'Left surface variance mean': var_stats_l[0],
        'Left surface variance std': var_stats_l[1],
        'Left surface variance min': var_stats_l[2],
        'Left surface variance max': var_stats_l[3],
        'Left surface dominant angle mean': dir_stats_l[0],
        'Left surface dominant angle std': dir_stats_l[1],
        'Right surface vector magnitude mean': right['mean_mag'],
        'Right surface variance (current)': right['var'],
        'Right surface variance mean': var_stats_r[0],
        'Right surface variance std': var_stats_r[1],
        'Right surface variance min': var_stats_r[2],
        'Right surface variance max': var_stats_r[3],
        'Right surface dominant angle mean': dir_stats_r[0],
        'Right surface dominant angle std': dir_stats_r[1],
    }

def compute_cheek_features(landmarks, prev_landmarks):
    if landmarks is None: return {}
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    left_ys = [norm_landmarks[i][1] for i in left_cheek_idx if i < len(norm_landmarks)]
    left_xs = [norm_landmarks[i][0] for i in left_cheek_idx if i < len(norm_landmarks)]
    right_ys = [norm_landmarks[i][1] for i in right_cheek_idx if i < len(norm_landmarks)]
    right_xs = [norm_landmarks[i][0] for i in right_cheek_idx if i < len(norm_landmarks)]

    left_raise = -np.mean(left_ys) + np.mean(left_xs) if left_ys else 0
    right_raise = -np.mean(right_ys) + np.mean(right_xs) if right_ys else 0
    overall = (left_raise + right_raise) / 2

    cheek_raise_buffer.append(overall)
    stats = [np.mean(cheek_raise_buffer), np.std(cheek_raise_buffer)] if len(cheek_raise_buffer) > 1 else [0, 0]
    vel = abs(overall - cheek_raise_buffer[-2]) if len(cheek_raise_buffer) > 1 else 0
    cheek_vel_buffer.append(vel)
    vel_stats = [np.mean(cheek_vel_buffer), np.std(cheek_vel_buffer)] if len(cheek_vel_buffer) > 1 else [0, 0]
    rapid = len(find_peaks(list(cheek_vel_buffer), distance=2)[0]) if len(cheek_vel_buffer) > 1 else 0
    var = np.var(cheek_raise_buffer) if len(cheek_raise_buffer) > 1 else 0
    freq = np.mean(np.abs(fft(list(cheek_raise_buffer)))[:buffer_size//2]) if len(cheek_raise_buffer) == buffer_size else 0

    asym = abs(left_raise - right_raise)
    surface = compute_surface_vectors_split(landmarks, prev_landmarks, left_cheek_idx_surface, right_cheek_idx_surface)
    l, r = surface['left'], surface['right']
    cheek_surface_var_buffer.append({'left': l['var'], 'right': r['var']})
    cheek_surface_dir_buffer.append({'left': l['angle'], 'right': r['angle']})

    lv = [x['left'] for x in list(cheek_surface_var_buffer)[-10:]]
    rv = [x['right'] for x in list(cheek_surface_var_buffer)[-10:]]
    la = [x['left'] for x in list(cheek_surface_dir_buffer)[-10:]]
    ra = [x['right'] for x in list(cheek_surface_dir_buffer)[-10:]]

    vl = [np.mean(lv), np.std(lv), np.min(lv), np.max(lv)] if lv else [0]*4
    vr = [np.mean(rv), np.std(rv), np.min(rv), np.max(rv)] if rv else [0]*4
    dl = [np.mean(la), np.std(la)] if len(la) > 1 else [0, 0]
    dr = [np.mean(ra), np.std(ra)] if len(ra) > 1 else [0, 0]

    return {
        'Cheek puff micro-expression variance mean': var,
        'Cheek puff rapid changes count': rapid,
        'Cheek raise (mean)': stats[0],
        'Cheek raise (std)': stats[1],
        'Cheek velocity (mean)': vel_stats[0],
        'Cheek velocity (std)': vel_stats[1],
        'Cheek frequency mean': freq,
        'Cheek asymmetry (mean)': asym,
        'Left surface vector magnitude mean': l['mean_mag'],
        'Left surface variance (current)': l['var'],
        'Left surface variance mean': vl[0],
        'Left surface variance std': vl[1],
        'Left surface variance min': vl[2],
        'Left surface variance max': vl[3],
        'Left surface dominant angle mean': dl[0],
        'Left surface dominant angle std': dl[1],
        'Right surface vector magnitude mean': r['mean_mag'],
        'Right surface variance (current)': r['var'],
        'Right surface variance mean': vr[0],
        'Right surface variance std': vr[1],
        'Right surface variance min': vr[2],
        'Right surface variance max': vr[3],
        'Right surface dominant angle mean': dr[0],
        'Right surface dominant angle std': dr[1],
    }

def compute_eye_features(landmarks, prev_landmarks):
    if landmarks is None: return {}
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    lu = (norm_landmarks[159][1] + norm_landmarks[158][1] + norm_landmarks[160][1]) / 3
    ll = (norm_landmarks[145][1] + norm_landmarks[144][1] + norm_landmarks[153][1]) / 3
    lw = abs(norm_landmarks[33][0] - norm_landmarks[133][0])
    left_ratio = abs(lu - ll) / lw if lw > 0 else 0

    ru = (norm_landmarks[386][1] + norm_landmarks[387][1] + norm_landmarks[385][1]) / 3
    rl = (norm_landmarks[374][1] + norm_landmarks[373][1] + norm_landmarks[380][1]) / 3
    rw = abs(norm_landmarks[362][0] - norm_landmarks[263][0])
    right_ratio = abs(ru - rl) / rw if rw > 0 else 0

    ratio = (left_ratio + right_ratio) / 2
    eye_ratio_buffer.append(ratio)
    stats = [np.mean(eye_ratio_buffer), np.std(eye_ratio_buffer)] if len(eye_ratio_buffer) > 1 else [0, 0]
    vel = abs(ratio - eye_ratio_buffer[-2]) if len(eye_ratio_buffer) > 1 else 0
    eye_vel_buffer.append(vel)
    vel_stats = [np.mean(eye_vel_buffer), np.std(eye_vel_buffer)] if len(eye_vel_buffer) > 1 else [0, 0]
    rapid = len(find_peaks(list(eye_vel_buffer), distance=2)[0]) if len(eye_vel_buffer) > 1 else 0
    var = np.var(eye_ratio_buffer) if len(eye_ratio_buffer) > 1 else 0
    blink = 1 if ratio < 0.1 else 0
    blink_buffer.append(blink)
    blink_rate = sum(blink_buffer) / len(blink_buffer) if blink_buffer else 0

    surface = compute_surface_vectors_split(landmarks, prev_landmarks, left_eye_idx_surface, right_eye_idx_surface)
    l, r = surface['left'], surface['right']
    eye_surface_var_buffer.append({'left': l['var'], 'right': r['var']})
    eye_surface_dir_buffer.append({'left': l['angle'], 'right': r['angle']})

    lv = [x['left'] for x in list(eye_surface_var_buffer)[-10:]]
    rv = [x['right'] for x in list(eye_surface_var_buffer)[-10:]]
    la = [x['left'] for x in list(eye_surface_dir_buffer)[-10:]]
    ra = [x['right'] for x in list(eye_surface_dir_buffer)[-10:]]

    vl = [np.mean(lv), np.std(lv), np.min(lv), np.max(lv)] if lv else [0]*4
    vr = [np.mean(rv), np.std(rv), np.min(rv), np.max(rv)] if rv else [0]*4
    dl = [np.mean(la), np.std(la)] if len(la) > 1 else [0, 0]
    dr = [np.mean(ra), np.std(ra)] if len(ra) > 1 else [0, 0]

    return {
        'Eye widening micro-expression variance mean': var,
        'Eye widening rapid changes count': rapid,
        'Eye ratio (mean)': stats[0],
        'Eye ratio (std)': stats[1],
        'Blink rate': blink_rate,
        'Eye squint velocity (mean)': vel_stats[0],
        'Eye squint velocity (std)': vel_stats[1],
        'Left surface vector magnitude mean': l['mean_mag'],
        'Left surface variance (current)': l['var'],
        'Left surface variance mean': vl[0],
        'Left surface variance std': vl[1],
        'Left surface variance min': vl[2],
        'Left surface variance max': vl[3],
        'Left surface dominant angle mean': dl[0],
        'Left surface dominant angle std': dl[1],
        'Right surface vector magnitude mean': r['mean_mag'],
        'Right surface variance (current)': r['var'],
        'Right surface variance mean': vr[0],
        'Right surface variance std': vr[1],
        'Right surface variance min': vr[2],
        'Right surface variance max': vr[3],
        'Right surface dominant angle mean': dr[0],
        'Right surface dominant angle std': dr[1],
    }

def compute_jaw_features(landmarks, prev_landmarks):
    if landmarks is None: return {}
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    chin = norm_landmarks[152]
    upper_jaw_ref = norm_landmarks[13]
    jaw_open = np.linalg.norm(chin - upper_jaw_ref)
    jaw_open_buffer.append(jaw_open)
    jaw_open_stats = [np.mean(jaw_open_buffer), np.std(jaw_open_buffer), np.min(jaw_open_buffer), np.max(jaw_open_buffer)] if len(jaw_open_buffer) > 1 else [0.0, 0.0, 0.0, 0.0]

    jaw_vel = abs(jaw_open - jaw_open_buffer[-2]) if len(jaw_open_buffer) > 1 else 0
    jaw_vel_buffer.append(jaw_vel)
    jaw_vel_stats = [np.mean(jaw_vel_buffer), np.std(jaw_vel_buffer)] if len(jaw_vel_buffer) > 1 else [0.0, 0.0]

    left_jaw = norm_landmarks[136]
    right_jaw = norm_landmarks[400]
    jaw_asym = np.abs(left_jaw[0] - right_jaw[0])
    jaw_asym_stats = [np.mean([jaw_asym]), np.std([jaw_asym]), np.max([jaw_asym])] if len(jaw_open_buffer) > 1 else [0.0, 0.0, 0.0]

    rapid_count = len(find_peaks(list(jaw_vel_buffer), distance=2)[0]) if len(jaw_vel_buffer) > 1 else 0
    sig_mov_count = sum(1 for v in jaw_vel_buffer if v > 0.001)

    freq_mean = np.mean(np.abs(fft(list(jaw_open_buffer)))[:buffer_size//2]) if len(jaw_open_buffer) == buffer_size else 0.0
    peak_freq = np.max(np.abs(fft(list(jaw_open_buffer)))[:buffer_size//2]) if len(jaw_open_buffer) == buffer_size else 0.0

    surface = compute_surface_vectors_split(landmarks, prev_landmarks, left_jaw_idx_surface, right_jaw_idx_surface)
    l, r = surface['left'], surface['right']
    jaw_surface_var_buffer.append({'left': l['var'], 'right': r['var']})
    jaw_surface_dir_buffer.append({'left': l['angle'], 'right': r['angle']})

    lv = [x['left'] for x in list(jaw_surface_var_buffer)[-10:]]
    rv = [x['right'] for x in list(jaw_surface_var_buffer)[-10:]]
    la = [x['left'] for x in list(jaw_surface_dir_buffer)[-10:]]
    ra = [x['right'] for x in list(jaw_surface_dir_buffer)[-10:]]

    vl = [np.mean(lv), np.std(lv), np.min(lv), np.max(lv)] if lv else [0]*4
    vr = [np.mean(rv), np.std(rv), np.min(rv), np.max(rv)] if rv else [0]*4
    dl = [np.mean(la), np.std(la)] if len(la) > 1 else [0, 0]
    dr = [np.mean(ra), np.std(ra)] if len(ra) > 1 else [0, 0]

    return {
        'Jaw opening (mean)': jaw_open_stats[0],
        'Jaw opening (std)': jaw_open_stats[1],
        'Jaw opening (min)': jaw_open_stats[2],
        'Jaw opening (max)': jaw_open_stats[3],
        'Jaw velocity (mean)': jaw_vel_stats[0],
        'Jaw velocity (std)': jaw_vel_stats[1],
        'Jaw asymmetry (mean)': jaw_asym_stats[0],
        'Jaw asymmetry (std)': jaw_asym_stats[1],
        'Jaw asymmetry (max)': jaw_asym_stats[2],
        'Jaw rapid changes count': rapid_count,
        'Jaw significant movements count': sig_mov_count,
        'Jaw frequency mean': freq_mean,
        'Jaw peak frequency': peak_freq,
        'Left surface vector magnitude mean': l['mean_mag'],
        'Left surface variance (current)': l['var'],
        'Left surface variance mean': vl[0],
        'Left surface variance std': vl[1],
        'Left surface variance min': vl[2],
        'Left surface variance max': vl[3],
        'Left surface dominant angle mean': dl[0],
        'Left surface dominant angle std': dl[1],
        'Right surface vector magnitude mean': r['mean_mag'],
        'Right surface variance (current)': r['var'],
        'Right surface variance mean': vr[0],
        'Right surface variance std': vr[1],
        'Right surface variance min': vr[2],
        'Right surface variance max': vr[3],
        'Right surface dominant angle mean': dr[0],
        'Right surface dominant angle std': dr[1],
    }

def compute_lips_features(landmarks, prev_landmarks):
    if landmarks is None: return {}
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    upper_lip = norm_landmarks[13]
    lower_lip = norm_landmarks[14]
    lip_open = np.linalg.norm(upper_lip - lower_lip)
    lips_open_buffer.append(lip_open)
    lip_open_stats = [np.mean(lips_open_buffer), np.std(lips_open_buffer), np.min(lips_open_buffer), np.max(lips_open_buffer)] if len(lips_open_buffer) > 1 else [0.0, 0.0, 0.0, 0.0]

    lip_vel = abs(lip_open - lips_open_buffer[-2]) if len(lips_open_buffer) > 1 else 0
    lips_vel_buffer.append(lip_vel)
    lip_vel_stats = [np.mean(lips_vel_buffer), np.std(lips_vel_buffer)] if len(lips_vel_buffer) > 1 else [0.0, 0.0]

    micro_var = np.var(lips_open_buffer) if len(lips_open_buffer) > 1 else 0.0
    rapid_count = len(find_peaks(list(lips_vel_buffer), distance=2)[0]) if len(lips_vel_buffer) > 1 else 0
    sig_mov_count = sum(1 for v in lips_vel_buffer if v > 0.001)

    freq_mean = np.mean(np.abs(fft(list(lips_open_buffer)))[:buffer_size//2]) if len(lips_open_buffer) == buffer_size else 0.0
    peak_freq = np.max(np.abs(fft(list(lips_open_buffer)))[:buffer_size//2]) if len(lips_open_buffer) == buffer_size else 0.0

    left_corner_y = norm_landmarks[61][1] - norm_landmarks[17][1]
    right_corner_y = norm_landmarks[291][1] - norm_landmarks[17][1]
    corner_asym = np.abs(left_corner_y - right_corner_y)
    corner_asym_stats = [np.mean([corner_asym]), np.std([corner_asym]), np.max([corner_asym])] if len(lips_open_buffer) > 1 else [0.0, 0.0, 0.0]

    surface = compute_surface_vectors_split(landmarks, prev_landmarks, left_lip_idx_surface, right_lip_idx_surface)
    l, r = surface['left'], surface['right']
    lips_surface_var_buffer.append({'left': l['var'], 'right': r['var']})
    lips_surface_dir_buffer.append({'left': l['angle'], 'right': r['angle']})

    lv = [x['left'] for x in list(lips_surface_var_buffer)[-10:]]
    rv = [x['right'] for x in list(lips_surface_var_buffer)[-10:]]
    la = [x['left'] for x in list(lips_surface_dir_buffer)[-10:]]
    ra = [x['right'] for x in list(lips_surface_dir_buffer)[-10:]]

    vl = [np.mean(lv), np.std(lv), np.min(lv), np.max(lv)] if lv else [0]*4
    vr = [np.mean(rv), np.std(rv), np.min(rv), np.max(rv)] if rv else [0]*4
    dl = [np.mean(la), np.std(la)] if len(la) > 1 else [0, 0]
    dr = [np.mean(ra), np.std(ra)] if len(ra) > 1 else [0, 0]

    return {
        'Lip micro-expression variance mean': micro_var,
        'Lip micro-expression rapid changes count': rapid_count,
        'Lip opening (mean)': lip_open_stats[0],
        'Lip opening (std)': lip_open_stats[1],
        'Lip opening (min)': lip_open_stats[2],
        'Lip opening (max)': lip_open_stats[3],
        'Lip velocity (mean)': lip_vel_stats[0],
        'Lip velocity (std)': lip_vel_stats[1],
        'Lip significant movements count': sig_mov_count,
        'Lip frequency mean': freq_mean,
        'Lip peak frequency': peak_freq,
        'Lip corner asymmetry (mean)': corner_asym_stats[0],
        'Lip corner asymmetry (std)': corner_asym_stats[1],
        'Lip corner asymmetry (max)': corner_asym_stats[2],
        'Left surface vector magnitude mean': l['mean_mag'],
        'Left surface variance (current)': l['var'],
        'Left surface variance mean': vl[0],
        'Left surface variance std': vl[1],
        'Left surface variance min': vl[2],
        'Left surface variance max': vl[3],
        'Left surface dominant angle mean': dl[0],
        'Left surface dominant angle std': dl[1],
        'Right surface vector magnitude mean': r['mean_mag'],
        'Right surface variance (current)': r['var'],
        'Right surface variance mean': vr[0],
        'Right surface variance std': vr[1],
        'Right surface variance min': vr[2],
        'Right surface variance max': vr[3],
        'Right surface dominant angle mean': dr[0],
        'Right surface dominant angle std': dr[1],
    }

def compute_mouth_features(landmarks, prev_landmarks):
    if landmarks is None: return {}
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    upper_lip = norm_landmarks[13]
    lower_lip = norm_landmarks[14]
    mouth_open = np.linalg.norm(upper_lip - lower_lip)
    mouth_open_buffer.append(mouth_open)
    mouth_open_stats = [np.mean(mouth_open_buffer), np.std(mouth_open_buffer), np.min(mouth_open_buffer), np.max(mouth_open_buffer)] if len(mouth_open_buffer) > 1 else [0.0, 0.0, 0.0, 0.0]

    mouth_vel = abs(mouth_open - mouth_open_buffer[-2]) if len(mouth_open_buffer) > 1 else 0
    mouth_vel_buffer.append(mouth_vel)
    mouth_vel_stats = [np.mean(mouth_vel_buffer), np.std(mouth_vel_buffer)] if len(mouth_vel_buffer) > 1 else [0.0, 0.0]

    micro_var = np.var(mouth_open_buffer) if len(mouth_open_buffer) > 1 else 0.0
    rapid_count = len(find_peaks(list(mouth_vel_buffer), distance=2)[0]) if len(mouth_vel_buffer) > 1 else 0
    sig_mov_count = sum(1 for v in mouth_vel_buffer if v > 0.001)

    freq_mean = np.mean(np.abs(fft(list(mouth_open_buffer)))[:buffer_size//2]) if len(mouth_open_buffer) == buffer_size else 0.0
    peak_freq = np.max(np.abs(fft(list(mouth_open_buffer)))[:buffer_size//2]) if len(mouth_open_buffer) == buffer_size else 0.0

    left_corner_y = norm_landmarks[61][1] - norm_landmarks[17][1]
    right_corner_y = norm_landmarks[291][1] - norm_landmarks[17][1]
    corner_asym = np.abs(left_corner_y - right_corner_y)
    corner_asym_stats = [np.mean([corner_asym]), np.std([corner_asym]), np.max([corner_asym])] if len(mouth_open_buffer) > 1 else [0.0, 0.0, 0.0]

    surface = compute_surface_vectors_split(landmarks, prev_landmarks, left_mouth_idx_surface, right_mouth_idx_surface)
    l, r = surface['left'], surface['right']
    mouth_surface_var_buffer.append({'left': l['var'], 'right': r['var']})
    mouth_surface_dir_buffer.append({'left': l['angle'], 'right': r['angle']})

    lv = [x['left'] for x in list(mouth_surface_var_buffer)[-10:]]
    rv = [x['right'] for x in list(mouth_surface_var_buffer)[-10:]]
    la = [x['left'] for x in list(mouth_surface_dir_buffer)[-10:]]
    ra = [x['right'] for x in list(mouth_surface_dir_buffer)[-10:]]

    vl = [np.mean(lv), np.std(lv), np.min(lv), np.max(lv)] if lv else [0]*4
    vr = [np.mean(rv), np.std(rv), np.min(rv), np.max(rv)] if rv else [0]*4
    dl = [np.mean(la), np.std(la)] if len(la) > 1 else [0, 0]
    dr = [np.mean(ra), np.std(ra)] if len(ra) > 1 else [0, 0]

    return {
        'Mouth micro-expression variance mean': micro_var,
        'Mouth micro-expression rapid changes count': rapid_count,
        'Mouth opening (mean)': mouth_open_stats[0],
        'Mouth opening (std)': mouth_open_stats[1],
        'Mouth opening (min)': mouth_open_stats[2],
        'Mouth opening (max)': mouth_open_stats[3],
        'Mouth velocity (mean)': mouth_vel_stats[0],
        'Mouth velocity (std)': mouth_vel_stats[1],
        'Mouth significant movements count': sig_mov_count,
        'Mouth frequency mean': freq_mean,
        'Mouth peak frequency': peak_freq,
        'Mouth corner asymmetry (mean)': corner_asym_stats[0],
        'Mouth corner asymmetry (std)': corner_asym_stats[1],
        'Mouth corner asymmetry (max)': corner_asym_stats[2],
        'Left surface vector magnitude mean': l['mean_mag'],
        'Left surface variance (current)': l['var'],
        'Left surface variance mean': vl[0],
        'Left surface variance std': vl[1],
        'Left surface variance min': vl[2],
        'Left surface variance max': vl[3],
        'Left surface dominant angle mean': dl[0],
        'Left surface dominant angle std': dl[1],
        'Right surface vector magnitude mean': r['mean_mag'],
        'Right surface variance (current)': r['var'],
        'Right surface variance mean': vr[0],
        'Right surface variance std': vr[1],
        'Right surface variance min': vr[2],
        'Right surface variance max': vr[3],
        'Right surface dominant angle mean': dr[0],
        'Right surface dominant angle std': dr[1],
    }

# ==================== Video Processing ====================
def process_video():
    global frame_global, features_global, prev_landmarks_global, recording, recorded_features, recording_frame_count
    cap = cv2.VideoCapture(0)
    prev_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        landmarks = None

        if results.multi_face_landmarks:
            lmks = results.multi_face_landmarks[0].landmark
            landmarks = [[lm.x, lm.y, lm.z] for lm in lmks]
            h, w, _ = frame.shape

            # Draw key landmarks
            for idx in set(brow_landmarks_idx + cheek_landmarks_idx + eye_landmarks_idx + jaw_landmarks_idx + lip_landmarks_idx + mouth_landmarks_idx):
                if idx < len(lmks):
                    lm = lmks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

        # Update features
        features_global = {
            'Brow': compute_brow_features(landmarks, prev_landmarks),
            'Cheek': compute_cheek_features(landmarks, prev_landmarks),
            'Eye': compute_eye_features(landmarks, prev_landmarks),
            'Jaw': compute_jaw_features(landmarks, prev_landmarks),
            'Lips': compute_lips_features(landmarks, prev_landmarks),
            'Mouth': compute_mouth_features(landmarks, prev_landmarks),
        }

        # Record if recording is active
        # Skip first 10 frames to ensure buffers have enough data for proper feature computation
        if recording:
            recording_frame_count += 1
            # Only record if we have valid landmarks (face detected) and after skipping initial frames
            if landmarks is not None and recording_frame_count > 10:
                recorded_features.append(features_global.copy())
        else:
            recording_frame_count = 0

        prev_landmarks = landmarks
        frame_global = frame.copy()
        time.sleep(0.03)

    cap.release()

# ==================== Flask App ====================
app = Flask(__name__)

def gen_frames():
    while True:
        if frame_global is not None:
            ret, buf = cv2.imencode('.jpg', frame_global)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/features')
def features():
    return jsonify(features_global)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, recorded_features, prev_landmarks_global, recording_frame_count
    
    # Clear recorded features for fresh recording
    recorded_features = []
    recording_frame_count = 0  # Reset frame counter
    
    # Note: Buffers are not cleared here because camera runs continuously
    # They should already have history from continuous processing
    # If you want to clear buffers, uncomment the following:
    # for buffer in [brow_raise_buffer, brow_left_raise_buffer, brow_right_raise_buffer, 
    #                brow_inner_raise_buffer, brow_vel_buffer, brow_surface_var_buffer, brow_surface_dir_buffer,
    #                cheek_raise_buffer, cheek_vel_buffer, cheek_surface_var_buffer, cheek_surface_dir_buffer,
    #                eye_ratio_buffer, eye_vel_buffer, blink_buffer, eye_surface_var_buffer, eye_surface_dir_buffer,
    #                jaw_open_buffer, jaw_vel_buffer, jaw_surface_var_buffer, jaw_surface_dir_buffer,
    #                lips_open_buffer, lips_vel_buffer, lips_surface_var_buffer, lips_surface_dir_buffer,
    #                mouth_open_buffer, mouth_vel_buffer, mouth_surface_var_buffer, mouth_surface_dir_buffer]:
    #     buffer.clear()
    # prev_landmarks_global = None
    
    recording = True
    return jsonify({"status": "Recording started"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    return jsonify({"status": "Recording stopped", "frames_recorded": len(recorded_features)})

@app.route('/predict', methods=['POST'])
def predict():
    global recorded_features, trained_model, label_encoder
    
    if not recorded_features:
        return jsonify({"error": "No data recorded. Please record first.", "emotion": None})
    
    if trained_model is None or label_encoder is None:
        return jsonify({"error": "Model not loaded. Please ensure best_model.pkl and label_encoder.pkl exist.", "emotion": None})
    
    # Require minimum frames
    min_frames = 30
    if len(recorded_features) < min_frames:
        return jsonify({
            "error": f"Only {len(recorded_features)} frames recorded. Need at least {min_frames} frames for accurate prediction. Please record for at least 2-3 seconds.",
            "emotion": None
        })
    
    try:
        # Flatten and average features
        flat_list = []
        for feats in recorded_features:
            flat = {}
            for region, values in feats.items():
                for k, v in values.items():
                    flat[f"{region}_{k}"] = v
            flat_list.append(flat)
        
        df = pd.DataFrame(flat_list).fillna(0)
        
        # Check if features are too uniform
        feature_std = df.std()
        feature_mean = df.mean()
        
        if feature_std.sum() < 0.001:
            return jsonify({
                "error": f"Features appear too uniform (std={feature_std.sum():.6f}). Make sure you're moving your face and expressing emotions CLEARLY during recording. Try making more EXAGGERATED expressions.",
                "emotion": None
            })
        
        # Check if most features are zero
        non_zero_features = (np.abs(feature_mean) > 0.0001).sum()
        total_features = len(feature_mean)
        if non_zero_features < total_features * 0.1:
            return jsonify({
                "error": f"Too many zero features ({non_zero_features}/{total_features}). Buffers may not have filled properly. Try recording longer (4-5 seconds).",
                "emotion": None
            })
        
        # Load training features to match column order
        try:
            train_df = pd.read_csv("facial_features.csv")
            train_columns = [col for col in train_df.columns if col != 'label']
            
            # Reorder and fill missing columns
            for col in train_columns:
                if col not in df.columns:
                    df[col] = 0
            
            df = df[train_columns]  # Reorder to match training
        except Exception as e:
            return jsonify({
                "error": f"Could not load training data columns: {str(e)}",
                "emotion": None
            })
        
        # ✅ CRITICAL FIX #1: Use StandardScaler (same as training)
        if feature_scaler is None:
            return jsonify({
                "error": "Feature scaler not loaded. Please retrain the model to generate feature_scaler.pkl",
                "emotion": None
            })
        
        # Scale features using the same scaler used during training
        df_scaled = pd.DataFrame(
            feature_scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        
        # ✅ CRITICAL FIX #2: Temporal smoothing - predict on each frame, then average probabilities
        # This reduces noise from single-frame predictions
        TEMPORAL_WINDOW = min(10, len(df_scaled))  # Use last 10 frames or all if less
        frame_probabilities = []
        
        # Predict on each frame (or use sliding window)
        frames_to_use = df_scaled.tail(TEMPORAL_WINDOW)  # Use last N frames
        
        for idx, row in frames_to_use.iterrows():
            frame_features = row.values.reshape(1, -1)
            frame_proba = trained_model.predict_proba(frame_features)[0]
            frame_probabilities.append(frame_proba)
        
        # Average probabilities across frames (temporal smoothing)
        avg_proba = np.mean(frame_probabilities, axis=0)
        
        # Get emotion classes
        emotion_classes = label_encoder.classes_
        all_probs = {emotion_classes[i]: float(avg_proba[i]) for i in range(len(emotion_classes))}
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Get top prediction
        top_emotion_idx = np.argmax(avg_proba)
        emotion = emotion_classes[top_emotion_idx]
        confidence = float(avg_proba[top_emotion_idx])
        
        # ✅ CRITICAL FIX #3: Confidence threshold - reject low confidence predictions
        CONFIDENCE_THRESHOLD = 0.25  # Minimum 25% confidence required
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "error": f"Low confidence prediction ({confidence:.1%}). Model is uncertain. Try making a more CLEAR and EXAGGERATED expression. Top prediction: {emotion} ({confidence:.1%})",
                "emotion": None,
                "confidence": confidence,
                "all_probabilities": all_probs
            })
        
        bias_corrected = False  # No longer needed with proper scaling
        
        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "frames_used": len(recorded_features),
            "bias_corrected": bias_corrected,
            "original_prediction": top_emotion if bias_corrected else None
        })
    except Exception as e:
        return jsonify({"error": str(e), "emotion": None})

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Facial Emotion Detection</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-align: center;
                padding: 20px;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .buttons {
                margin: 30px 0;
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 15px;
            }
            .button {
                padding: 15px 35px;
                font-size: 18px;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .button:active {
                transform: translateY(0);
            }
            .open { background: #90ee90; color: #000; }
            .start { background: #4CAF50; color: white; }
            .stop { background: #f44336; color: white; }
            .submit { background: #2196F3; color: white; }
            .button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            #cameraSpace {
                width: 100%;
                max-width: 800px;
                height: 600px;
                margin: 20px auto;
                background: #000;
                border-radius: 15px;
                overflow: hidden;
                display: none;
                box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            }
            #cameraSpace img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            #result {
                font-size: 2.5em;
                font-weight: bold;
                margin: 40px 0;
                min-height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                border-radius: 15px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .result-emotion {
                color: #2196F3;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .result-confidence {
                font-size: 0.6em;
                color: #666;
                margin-top: 10px;
            }
            .status {
                padding: 10px 20px;
                border-radius: 25px;
                display: inline-block;
                margin: 10px;
                font-weight: bold;
            }
            .status.recording {
                background: #f44336;
                color: white;
                animation: pulse 1.5s infinite;
            }
            .status.ready {
                background: #4CAF50;
                color: white;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            .biomarker-container {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 30px;
            }
            .biomarker-box {
                width: 380px;
                padding: 20px;
                background: #f8f9fa;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                border-radius: 15px;
                max-height: 600px;
                overflow-y: auto;
            }
            .biomarker-box h3 {
                margin-top: 0;
                text-align: center;
                color: #333;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .biomarker-box h4 {
                color: #667eea;
                margin-top: 15px;
                margin-bottom: 8px;
                font-size: 1em;
            }
            .biomarker-box p {
                margin: 5px 0;
                font-size: 0.9em;
                color: #555;
                padding: 3px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Facial Emotion Detection</h1>
            <p class="subtitle">Real-time emotion recognition using facial micro-features</p>
            
            <div class="buttons">
                <button class="button open" onclick="openCamera()">📷 Open Camera</button>
                <button class="button start" id="startBtn" onclick="startRec()" disabled>▶️ Start Recording</button>
                <button class="button stop" id="stopBtn" onclick="stopRec()" disabled>⏹️ Stop Recording</button>
                <button class="button submit" id="submitBtn" onclick="predictEmotion()" disabled>✅ Submit & Predict</button>
            </div>
            
            <div id="cameraSpace">
                <img id="video" src="/video_feed">
            </div>
            
            <div id="result">
                <div>
                    <div class="status ready" id="status">Ready</div>
                    <div id="resultText">Click 'Open Camera' to begin</div>
                </div>
            </div>

            <div class="biomarker-container">
                <div class="biomarker-box"><h3>👁️ Eye</h3><div id="Eye"></div></div>
                <div class="biomarker-box"><h3>😐 Brow</h3><div id="Brow"></div></div>
                <div class="biomarker-box"><h3>😊 Cheek</h3><div id="Cheek"></div></div>
                <div class="biomarker-box"><h3>👄 Mouth</h3><div id="Mouth"></div></div>
                <div class="biomarker-box"><h3>💋 Lips</h3><div id="Lips"></div></div>
                <div class="biomarker-box"><h3>🦷 Jaw</h3><div id="Jaw"></div></div>
            </div>
        </div>

        <script>
            let intervalId;
            let isRecording = false;
            
            function openCamera() {
                document.getElementById('cameraSpace').style.display = 'block';
                document.getElementById('startBtn').disabled = false;
                document.getElementById('status').textContent = 'Camera Ready';
                document.getElementById('status').className = 'status ready';
                document.getElementById('resultText').textContent = 'Click "Start Recording" to begin';
                intervalId = setInterval(updateFeatures, 150);
            }
            
            function startRec() {
                fetch('/start_recording', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        isRecording = true;
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        document.getElementById('status').textContent = '🔴 Recording...';
                        document.getElementById('status').className = 'status recording';
                        document.getElementById('resultText').textContent = 'Recording in progress... Speak naturally!';
                    });
            }
            
            function stopRec() {
                fetch('/stop_recording', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        isRecording = false;
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        document.getElementById('submitBtn').disabled = false;
                        document.getElementById('status').textContent = 'Recording Stopped';
                        document.getElementById('status').className = 'status ready';
                        document.getElementById('resultText').textContent = `Recording stopped. ${data.frames_recorded} frames captured. Click "Submit & Predict" to analyze.`;
                    });
            }
            
            function predictEmotion() {
                document.getElementById('resultText').textContent = 'Processing... Please wait...';
                document.getElementById('submitBtn').disabled = true;
                
                fetch('/predict', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        if (data.error) {
                            document.getElementById('resultText').innerHTML = `<span style="color: #f44336;">❌ Error: ${data.error}</span>`;
                        } else {
                            const emotion = data.emotion;
                            const confidence = (data.confidence * 100).toFixed(1);
                            
                            let resultHTML = `<span class="result-emotion">${emotion.toUpperCase()}</span><br>
                                 <span class="result-confidence">Confidence: ${confidence}%</span>`;
                            
                            // Show bias correction notice if applied
                            if (data.bias_corrected && data.original_prediction) {
                                resultHTML += `<div style="background: rgba(255, 193, 7, 0.2); padding: 10px; border-radius: 8px; margin: 10px auto; max-width: 500px; font-size: 0.8em; color: #856404;">
                                    🔧 Bias correction applied: Model initially predicted "${data.original_prediction}" but was corrected to "${emotion}"
                                </div>`;
                            }
                            
                            document.getElementById('resultText').innerHTML = resultHTML;
                            
                            // Show all probabilities
                            let probText = '<div style="margin-top: 15px; font-size: 0.5em; text-align: left; max-width: 500px; margin-left: auto; margin-right: auto;">';
                            for (const [emo, prob] of Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1])) {
                                const barWidth = prob * 100;
                                const isOriginal = (data.bias_corrected && emo === data.original_prediction);
                                const isCorrected = (emo === emotion);
                                let barColor = '#ccc';
                                if (isCorrected) barColor = '#2196F3';
                                else if (isOriginal) barColor = '#ff9800'; // Orange for original (corrected away from)
                                
                                probText += `<div style="margin: 5px 0;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                        <span>${isOriginal ? '⚠️ ' : ''}${isCorrected ? '👉 ' : ''}${emo}</span>
                                        <span>${(prob * 100).toFixed(1)}%</span>
                                    </div>
                                    <div style="background: #e0e0e0; border-radius: 5px; height: 8px; overflow: hidden;">
                                        <div style="background: ${barColor}; height: 100%; width: ${barWidth}%; transition: width 0.3s;"></div>
                                    </div>
                                </div>`;
                            }
                            probText += '</div>';
                            document.getElementById('resultText').innerHTML += probText;
                        }
                        document.getElementById('submitBtn').disabled = false;
                    })
                    .catch(error => {
                        document.getElementById('resultText').innerHTML = `<span style="color: #f44336;">❌ Error: ${error.message}</span>`;
                        document.getElementById('submitBtn').disabled = false;
                    });
            }
            
            function updateFeatures() {
                fetch('/features')
                    .then(r => r.json())
                    .then(data => {
                        for (const [key, feats] of Object.entries(data)) {
                            const box = document.getElementById(key);
                            if (!box) continue;
                            box.innerHTML = '';
                            
                            const left = [], right = [], other = [];
                            for (const [k, v] of Object.entries(feats)) {
                                const val = parseFloat(v).toFixed(4);
                                if (k.includes('Left surface')) {
                                    left.push(k.replace('Left surface ', '') + ': ' + val);
                                } else if (k.includes('Right surface')) {
                                    right.push(k.replace('Right surface ', '') + ': ' + val);
                                } else {
                                    other.push(k + ': ' + val);
                                }
                            }
                            
                            const addSection = (title, arr) => {
                                if (arr.length > 0) {
                                    const h = document.createElement('h4');
                                    h.textContent = title;
                                    box.appendChild(h);
                                    arr.forEach(line => {
                                        const p = document.createElement('p');
                                        p.textContent = line;
                                        box.appendChild(p);
                                    });
                                }
                            };
                            
                            addSection('Left Surface', left);
                            addSection('Right Surface', right);
                            addSection('Other Features', other);
                        }
                    })
                    .catch(err => console.error('Error fetching features:', err));
            }
        </script>
    </body>
    </html>
    """)

# ==================== Main ====================
if __name__ == '__main__':
    print("="*60)
    print("FACIAL EMOTION DETECTION WEB APP")
    print("="*60)
    
    # Load model
    if load_model():
        print("\nStarting web server...")
        print("Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server\n")
        
        # Start video processing in background thread
        t = threading.Thread(target=process_video, daemon=True)
        t.start()
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    else:
        print("\n❌ Failed to load model. Please ensure best_model.pkl and label_encoder.pkl exist.")
        print("   Train the model first using expression.ipynb")

