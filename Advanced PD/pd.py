import os
os.environ['GLOG_minloglevel'] = '2' # Suppress MediaPipe/TensorFlow warnings
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
from collections import deque
from scipy.spatial import Delaunay
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# CUDA Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Options
DATASET_DIR = "PD Videos/Training"
OUTPUT_DIR = "."
ALL_FEATURES_CSV = "pd_features.csv"
SELECTED_FEATURES_CSV = "pd_selected_features.csv"
MODEL_FILE = "pd_detection_model.pth"
SEQ_LENGTH = 240
NUM_FOLDS = 5

# ================= GLOBAL BUFFERS =================
buffer_size = 60
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

# --- Landmark Indices ---
left_brow_idx = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
right_brow_idx = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
inner_brow_idx = [63, 293]
left_cheek_idx = [205, 206, 216, 204, 207, 114, 115, 116, 213, 214, 215]
right_cheek_idx = [425, 426, 436, 424, 427, 343, 344, 345, 433, 434, 435]
left_eye_idx = [33, 160, 158, 133, 153, 144, 145, 159]
right_eye_idx = [362, 385, 387, 263, 373, 374, 380, 386]
outer_eye_idx = [33, 133, 362, 263]
jaw_landmarks_idx = [152, 176, 136, 172, 397, 365, 366, 379, 400, 378, 377]
lip_landmarks_idx = [13, 14, 37, 39, 40, 61, 78, 80, 81, 82, 84, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 321, 324, 375, 402, 405, 409, 415]
mouth_landmarks_idx = [13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

# --- LEFT / RIGHT Surface Vector Splits ---
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

_prev_landmarks_global = None

def reset_buffers():
    global _prev_landmarks_global
    _prev_landmarks_global = None
    for b in [brow_raise_buffer, brow_left_raise_buffer, brow_right_raise_buffer, brow_inner_raise_buffer, brow_vel_buffer, brow_surface_var_buffer, brow_surface_dir_buffer,
              cheek_raise_buffer, cheek_vel_buffer, cheek_surface_var_buffer, cheek_surface_dir_buffer,
              eye_ratio_buffer, eye_vel_buffer, blink_buffer, eye_surface_var_buffer, eye_surface_dir_buffer,
              jaw_open_buffer, jaw_vel_buffer, jaw_surface_var_buffer, jaw_surface_dir_buffer,
              lips_open_buffer, lips_vel_buffer, lips_surface_var_buffer, lips_surface_dir_buffer,
              mouth_open_buffer, mouth_vel_buffer, mouth_surface_var_buffer, mouth_surface_dir_buffer]:
        b.clear()

# --- Data Augmentation ---
# --- Data Augmentation ---
def add_temporal_noise(seq, noise_level=0.01):
    noise = np.random.normal(0, noise_level, seq.shape)
    return seq + noise

def temporal_dropout(seq, dropout_rate=0.05):
    if np.random.rand() < 0.3:
        mask = np.random.rand(len(seq)) > dropout_rate
        if np.sum(mask) > 1:
            return seq[mask]
    return seq

def video_augmentation(image, flip=False, brightness=1.0, contrast=1.0, rotation=0, shear=0):
    if flip:
        image = cv2.flip(image, 1)
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 255 - 255)
    if rotation != 0 or shear != 0:
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
        shear_matrix = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
        
        # Expand dimensions to 3x3 to allow multiplication
        M_3x3 = np.vstack([M, [0, 0, 1]])
        shear_3x3 = np.vstack([shear_matrix, [0, 0, 1]])
        
        combined_3x3 = np.dot(M_3x3, shear_3x3)
        M = combined_3x3[:2, :] # Take back top 2 rows
        
        image = cv2.warpAffine(image, M, (cols, rows))
    return image

# --- Normalization ---
def normalize_for_rotation_distance(landmarks, prev_landmarks):
    if landmarks is None or len(landmarks) == 0:
        return landmarks
    
    nose_tip = np.array(landmarks[1])
    normalized = []
    for lm in landmarks:
        lm_arr = np.array(lm)
        dist = np.linalg.norm(lm_arr - nose_tip) + 1e-6
        normalized.append((lm_arr - nose_tip) / dist)
    return normalized

# --- Surface Vector Split Function ---
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
            # Explicit 2D cross product to avoid NumPy 2.0 deprecation warning
            # v1 = p2 - p1, v2 = p3 - p1. Cross = v1_x * v2_y - v1_y * v2_x
            v1 = points2d[i2] - points2d[i1]
            v2 = points2d[i3] - points2d[i1]
            area = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])
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

        vectors = np.array(triangle_vectors) if triangle_vectors else np.array([])
        norms = triangle_norms
        positions = [np.mean([curr_pos[i] for i in s], axis=0) for s in tri.simplices]

        return {'vectors': vectors, 'positions': positions, 'norms': norms, 'mean_mag': mean_mag, 'var': var, 'angle': angle}

    return {'left': process_side(left_idx), 'right': process_side(right_idx)}

# --- Tremor Feature Extraction Helper ---
def compute_tremor_features(vel_buffer, freq_range=(4,7), sample_rate=30):
    if len(vel_buffer) < buffer_size:
        return {'tremor_power': 0.0, 'dominant_freq': 0.0, 'tremor_index': 0.0}
    
    vel = np.array(vel_buffer)
    fft_vals = fft(vel)
    freqs = fftfreq(len(vel), 1/sample_rate)
    power = np.abs(fft_vals)**2
    
    tremor_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    tremor_power = np.sum(power[tremor_mask]) / np.sum(power) if np.sum(power) > 0 else 0.0
    dominant_freq = freqs[np.argmax(power)] if len(power) > 0 else 0.0
    tremor_index = tremor_power * np.std(vel)
    
    return {'tremor_power': tremor_power, 'dominant_freq': dominant_freq, 'tremor_index': tremor_index}

# --- BROW ---
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

    tremor = compute_tremor_features(brow_vel_buffer)

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
        'Brow tremor power': tremor['tremor_power'],
        'Brow dominant frequency': tremor['dominant_freq'],
        'Brow tremor index': tremor['tremor_index'],

        'Brow Left surface vector magnitude mean': left['mean_mag'],
        'Brow Left surface variance (current)': left['var'],
        'Brow Left surface variance mean': var_stats_l[0],
        'Brow Left surface variance std': var_stats_l[1],
        'Brow Left surface variance min': var_stats_l[2],
        'Brow Left surface variance max': var_stats_l[3],
        'Brow Left surface dominant angle mean': dir_stats_l[0],
        'Brow Left surface dominant angle std': dir_stats_l[1],

        'Brow Right surface vector magnitude mean': right['mean_mag'],
        'Brow Right surface variance (current)': right['var'],
        'Brow Right surface variance mean': var_stats_r[0],
        'Brow Right surface variance std': var_stats_r[1],
        'Brow Right surface variance min': var_stats_r[2],
        'Brow Right surface variance max': var_stats_r[3],
        'Brow Right surface dominant angle mean': dir_stats_r[0],
        'Brow Right surface dominant angle std': dir_stats_r[1],
    }

# --- CHEEK --- #
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

    tremor = compute_tremor_features(cheek_vel_buffer)

    return {
        'Cheek puff micro-expression variance mean': var,
        'Cheek puff rapid changes count': rapid,
        'Cheek raise (mean)': stats[0],
        'Cheek raise (std)': stats[1],
        'Cheek velocity (mean)': vel_stats[0],
        'Cheek velocity (std)': vel_stats[1],
        'Cheek frequency mean': freq,
        'Cheek asymmetry (mean)': asym,
        'Cheek tremor power': tremor['tremor_power'],
        'Cheek dominant frequency': tremor['dominant_freq'],
        'Cheek tremor index': tremor['tremor_index'],

        'Cheek Left surface vector magnitude mean': l['mean_mag'],
        'Cheek Left surface variance (current)': l['var'],
        'Cheek Left surface variance mean': vl[0],
        'Cheek Left surface variance std': vl[1],
        'Cheek Left surface variance min': vl[2],
        'Cheek Left surface variance max': vl[3],
        'Cheek Left surface dominant angle mean': dl[0],
        'Cheek Left surface dominant angle std': dl[1],

        'Cheek Right surface vector magnitude mean': r['mean_mag'],
        'Cheek Right surface variance mean': vr[0],
        'Cheek Right surface variance std': vr[1],
        'Cheek Right surface variance min': vr[2],
        'Cheek Right surface variance max': vr[3],
        'Cheek Right surface dominant angle mean': dr[0],
        'Cheek Right surface dominant angle std': dr[1],
    }

# --- EYE ---
def compute_eye_features(landmarks, prev_landmarks):
    landmarks = normalize_for_rotation_distance(landmarks, prev_landmarks)
    prev_landmarks = normalize_for_rotation_distance(prev_landmarks, None) if prev_landmarks else None
    
    if landmarks is None: return {}
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    lu = (norm_landmarks[159][1] + norm_landmarks[158][1] + norm_landmarks[160][1]) / 3
    ll = (norm_landmarks[145][1] + norm_landmarks[144][1] + norm_landmarks[153][1]) / 3
    lw = abs(norm_landmarks[33][0] - norm_landmarks[133][0])
    left_ratio = abs(lu - ll) / max(lw, 1e-6)

    ru = (norm_landmarks[386][1] + norm_landmarks[387][1] + norm_landmarks[385][1]) / 3
    rl = (norm_landmarks[374][1] + norm_landmarks[373][1] + norm_landmarks[380][1]) / 3
    rw = abs(norm_landmarks[362][0] - norm_landmarks[263][0])
    right_ratio = abs(ru - rl) / max(rw, 1e-6)

    ratio = (left_ratio + right_ratio) / 2
    eye_ratio_buffer.append(ratio)
    stats = [np.mean(eye_ratio_buffer), np.std(eye_ratio_buffer)] if len(eye_ratio_buffer) > 1 else [0, 0]
    vel = abs(ratio - eye_ratio_buffer[-2]) if len(eye_ratio_buffer) > 1 else 0
    eye_vel_buffer.append(vel)
    vel_stats = [np.mean(eye_vel_buffer), np.std(eye_vel_buffer)] if len(eye_vel_buffer) > 1 else [0, 0]
    rapid = len(find_peaks(list(eye_vel_buffer), distance=2)[0]) if len(eye_vel_buffer) > 1 else 0
    var = np.var(eye_ratio_buffer) if len(eye_ratio_buffer) > 1 else 0
    blink = 1 if ratio < 0.15 else 0
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

        'Eye Left surface vector magnitude mean': l['mean_mag'],
        'Eye Left surface variance (current)': l['var'],
        'Eye Left surface variance mean': vl[0],
        'Eye Left surface variance std': vl[1],
        'Eye Left surface variance min': vl[2],
        'Eye Left surface variance max': vl[3],
        'Eye Left surface dominant angle mean': dl[0],
        'Eye Left surface dominant angle std': dl[1],

        'Eye Right surface vector magnitude mean': r['mean_mag'],
        'Eye Right surface variance (current)': r['var'],
        'Eye Right surface variance mean': vr[0],
        'Eye Right surface variance std': vr[1],
        'Eye Right surface variance min': vr[2],
        'Eye Right surface variance max': vr[3],
        'Eye Right surface dominant angle mean': dr[0],
        'Eye Right surface dominant angle std': dr[1],
    }

# --- JAW ---
def compute_jaw_features(landmarks, prev_landmarks):
    landmarks = normalize_for_rotation_distance(landmarks, prev_landmarks)
    prev_landmarks = normalize_for_rotation_distance(prev_landmarks, None) if prev_landmarks else None
    
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

        'Jaw Left surface vector magnitude mean': l['mean_mag'],
        'Jaw Left surface variance (current)': l['var'],
        'Jaw Left surface variance mean': vl[0],
        'Jaw Left surface variance std': vl[1],
        'Jaw Left surface variance min': vl[2],
        'Jaw Left surface variance max': vl[3],
        'Jaw Left surface dominant angle mean': dl[0],
        'Jaw Left surface dominant angle std': dl[1],

        'Jaw Right surface vector magnitude mean': r['mean_mag'],
        'Jaw Right surface variance (current)': r['var'],
        'Jaw Right surface variance mean': vr[0],
        'Jaw Right surface variance std': vr[1],
        'Jaw Right surface variance min': vr[2],
        'Jaw Right surface variance max': vr[3],
        'Jaw Right surface dominant angle mean': dr[0],
        'Jaw Right surface dominant angle std': dr[1],
    }

# --- LIPS ---
def compute_lips_features(landmarks, prev_landmarks):
    landmarks = normalize_for_rotation_distance(landmarks, prev_landmarks)
    prev_landmarks = normalize_for_rotation_distance(prev_landmarks, None) if prev_landmarks else None
    
    # Normalized landmarks passed in
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

        'Lip Left surface vector magnitude mean': l['mean_mag'],
        'Lip Left surface variance (current)': l['var'],
        'Lip Left surface variance mean': vl[0],
        'Lip Left surface variance std': vl[1],
        'Lip Left surface variance min': vl[2],
        'Lip Left surface variance max': vl[3],
        'Lip Left surface dominant angle mean': dl[0],
        'Lip Left surface dominant angle std': dl[1],

        'Lip Right surface vector magnitude mean': r['mean_mag'],
        'Lip Right surface variance (current)': r['var'],
        'Lip Right surface variance mean': vr[0],
        'Lip Right surface variance std': vr[1],
        'Lip Right surface variance min': vr[2],
        'Lip Right surface variance max': vr[3],
        'Lip Right surface dominant angle mean': dr[0],
        'Lip Right surface dominant angle std': dr[1],
    }

# --- MOUTH ---
def compute_mouth_features(landmarks, prev_landmarks):
    landmarks = normalize_for_rotation_distance(landmarks, prev_landmarks)
    prev_landmarks = normalize_for_rotation_distance(prev_landmarks, None) if prev_landmarks else None
    
    # Normalized landmarks passed in
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

        'Mouth Left surface vector magnitude mean': l['mean_mag'],
        'Mouth Left surface variance (current)': l['var'],
        'Mouth Left surface variance mean': vl[0],
        'Mouth Left surface variance std': vl[1],
        'Mouth Left surface variance min': vl[2],
        'Mouth Left surface variance max': vl[3],
        'Mouth Left surface dominant angle mean': dl[0],
        'Mouth Left surface dominant angle std': dl[1],

        'Mouth Right surface vector magnitude mean': r['mean_mag'],
        'Mouth Right surface variance (current)': r['var'],
        'Mouth Right surface variance mean': vr[0],
        'Mouth Right surface variance std': vr[1],
        'Mouth Right surface variance min': vr[2],
        'Mouth Right surface variance max': vr[3],
        'Mouth Right surface dominant angle mean': dr[0],
        'Mouth Right surface dominant angle std': dr[1],
    }

# --- LIPS ---
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

        'Lip Left surface vector magnitude mean': l['mean_mag'],
        'Lip Left surface variance (current)': l['var'],
        'Lip Left surface variance mean': vl[0],
        'Lip Left surface variance std': vl[1],
        'Lip Left surface variance min': vl[2],
        'Lip Left surface variance max': vl[3],
        'Lip Left surface dominant angle mean': dl[0],
        'Lip Left surface dominant angle std': dl[1],

        'Lip Right surface vector magnitude mean': r['mean_mag'],
        'Lip Right surface variance (current)': r['var'],
        'Lip Right surface variance mean': vr[0],
        'Lip Right surface variance std': vr[1],
        'Lip Right surface variance min': vr[2],
        'Lip Right surface variance max': vr[3],
        'Lip Right surface dominant angle mean': dr[0],
        'Lip Right surface dominant angle std': dr[1],
    }

# --- MOUTH ---
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

        'Mouth Left surface vector magnitude mean': l['mean_mag'],
        'Mouth Left surface variance (current)': l['var'],
        'Mouth Left surface variance mean': vl[0],
        'Mouth Left surface variance std': vl[1],
        'Mouth Left surface variance min': vl[2],
        'Mouth Left surface variance max': vl[3],
        'Mouth Left surface dominant angle mean': dl[0],
        'Mouth Left surface dominant angle std': dl[1],

        'Mouth Right surface vector magnitude mean': r['mean_mag'],
        'Mouth Right surface variance (current)': r['var'],
        'Mouth Right surface variance mean': vr[0],
        'Mouth Right surface variance std': vr[1],
        'Mouth Right surface variance min': vr[2],
        'Mouth Right surface variance max': vr[3],
        'Mouth Right surface dominant angle mean': dr[0],
        'Mouth Right surface dominant angle std': dr[1],
    }

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.):
    if not sequences:
        return np.array([])
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    num_samples = len(sequences)
    sample = sequences[0]
    if isinstance(sample, np.ndarray):
        if sample.ndim > 1:
            feat_dim = sample.shape[1]
            output_shape = (num_samples, maxlen, feat_dim)
        else:
            output_shape = (num_samples, maxlen)
    elif isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], (list, np.ndarray)):
         feat_dim = len(sample[0])
         output_shape = (num_samples, maxlen, feat_dim)
    else:
        output_shape = (num_samples, maxlen)
    x = np.full(output_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s): continue
        s = np.asarray(s, dtype=dtype)
        if truncating == 'pre': trunc = s[-maxlen:]
        else: trunc = s[:maxlen]
        if padding == 'post': x[idx, :len(trunc)] = trunc
        else: x[idx, -len(trunc):] = trunc
    return x

def extract_data_from_videos():
    global _prev_landmarks_global
    print("Starting Feature Extraction...")
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return None

    data = []
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    # Setup MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Check if model exists
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please download it.")
        return None

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for category in classes:
        print(f"Processing category: {category}")
        video_files = glob.glob(os.path.join(DATASET_DIR, category, "*.mp4"))
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            
            # 6 Augmentations per video
            for aug_idx in range(6): 
                if aug_idx == 0:
                    rot, shr, brt = 0, 0, 1.0
                    suffix = ""
                else:
                    rot = np.random.uniform(-10, 10)
                    shr = np.random.uniform(-0.1, 0.1)
                    brt = np.random.uniform(0.8, 1.2)
                    suffix = f"_aug{aug_idx}"
                
                reset_buffers()
                cap = cv2.VideoCapture(video_path)
                
                frame_timestamp_ms = 0
                frame_count = 0
                
                # Need fps for timestamp calculation
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30.0 # Default fallback
                
                # Create/Reset Landmarker for each video stream to handle timestamp resets
                with FaceLandmarker.create_from_options(options) as landmarker:
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success: break
                        frame_count += 1
                        frame_timestamp_ms = int(frame_count * (1000 / fps))
                        
                        # Apply Augmentation
                        if aug_idx > 0:
                            image = video_augmentation(image, brightness=brt, rotation=rot, shear=shr)

                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        
                        try:
                            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                            
                            if detection_result.face_landmarks:
                                landmarks = detection_result.face_landmarks[0] 
                                lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                                
                                lm_list_norm = normalize_for_rotation_distance(lm_list, _prev_landmarks_global)
                                
                                features_dict = {}
                                features_dict.update(compute_brow_features(lm_list_norm, _prev_landmarks_global))
                                features_dict.update(compute_cheek_features(lm_list_norm, _prev_landmarks_global))
                                features_dict.update(compute_eye_features(lm_list_norm, _prev_landmarks_global))
                                features_dict.update(compute_jaw_features(lm_list_norm, _prev_landmarks_global))
                                features_dict.update(compute_lips_features(lm_list_norm, _prev_landmarks_global))
                                features_dict.update(compute_mouth_features(lm_list_norm, _prev_landmarks_global))
                                
                                features_dict['Video'] = video_name + suffix
                                features_dict['Label'] = category
                                features_dict['Frame'] = frame_count
                                data.append(features_dict)
                                
                                _prev_landmarks_global = lm_list_norm
                        except Exception as e:
                            print(f"Error processing frame {frame_count} of {video_name} (Aug {aug_idx}): {e}")
                            # Don't break, try next frame? 
                            # If landmarker is broken for this stream, it might be better to break.
                            # But usually it's just a timestamp glitch or empty detection.
                            pass

                cap.release()
            
    df = pd.DataFrame(data)
    df = df.fillna(0)
    csv_path = os.path.join(OUTPUT_DIR, ALL_FEATURES_CSV)
    df.to_csv(csv_path, index=False)
    print(f"Feature extraction complete. Saved to {csv_path}")
    return df

# --- PCA SELECTION ---
# --- FEATURE SELECTION (Helper) ---
def select_features_via_pca(X, feature_names):
    # Standardize internally just for PCA calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.99)
    pca.fit(X_scaled)
    
    selected_indices = set()
    components = pca.components_
    for i in range(components.shape[0]):
        # Get top 10 features contributing to this component
        top_indices = np.argsort(np.abs(components[i]))[-10:]
        for idx in top_indices: selected_indices.add(idx)
        
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features

# --- MODEL ---
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + self.shortcut(x))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        return torch.sum(F_loss)

class OptimizedCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3):
        super().__init__()
        self.noise_std = 0.05
        # Reduced hidden dim to 64 as requested, added layer
        self.cnn = nn.Sequential(
            ResidualConvBlock(input_dim, 64),
            nn.MaxPool1d(2),
            ResidualConvBlock(64, 128),
            nn.MaxPool1d(2),
            ResidualConvBlock(128, 128), # Added layer
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True, num_layers=num_layers, bidirectional=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x, lengths):
        if self.training: x = x + torch.randn_like(x) * self.noise_std
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        # Calculate valid lengths after CNN reduction (3 maxpools -> reduction by 2^3 = 8)
        # Clamp to ensure we don't exceed actual tensor size (handle edge cases)
        max_len = x.size(1)
        adj_len = (lengths.cpu() // 8).clamp(min=1, max=max_len)
        
        packed = pack_padded_sequence(x, adj_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = torch.mean(attn_out, dim=1)
        return self.fc(out)

class ExpressionDataset(Dataset):
    def __init__(self, features, labels, lengths):
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)  # Will move to device in training loop
        self.lengths = torch.tensor(lengths, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        seq = self.features[idx]
        seq = temporal_dropout(seq)
        seq = add_temporal_noise(seq)
        return torch.tensor(seq, dtype=torch.float32), self.labels[idx], torch.tensor(len(seq), dtype=torch.long)

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    # Move to GPU (lengths stays on CPU for pack_padded_sequence compatibility)
    return padded_sequences.to(device), labels.to(device), lengths

def train_model(df=None):
    print("Starting Model Training with Strict Train/Val Separation...")
    if df is None:
        csv_path = os.path.join(OUTPUT_DIR, ALL_FEATURES_CSV)
        if os.path.exists(csv_path): df = pd.read_csv(csv_path)
        else: 
            print("No features file found.")
            return

    # Initial Columns
    drop_cols = ['Video', 'Label', 'Frame']
    all_feature_names = [c for c in df.columns if c not in drop_cols]
    
    # Encode Labels
    unique_labels = sorted(df['Label'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    print(f"Classes: {label_map}")

    # Group by Video to ensure no data leakage across frames
    grouped = df.groupby('Video')
    video_names = []
    X_seq_all = [] # List of (seq_len, num_features) arrays
    y_seq_all = [] # List of labels
    
    for video, group in grouped:
        feats = group[all_feature_names].values
        label = label_map[group['Label'].iloc[0]]
        
        # Truncate/Pad handled later, but we need raw sequences first
        if len(feats) > SEQ_LENGTH:
            feats = feats[:SEQ_LENGTH]
            
        X_seq_all.append(feats)
        y_seq_all.append(label)
        video_names.append(video)

    X_seq_all = np.array(X_seq_all, dtype=object) # Ragged array
    y_seq_all = np.array(y_seq_all)

    # Cross Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_accs = []
    
    # Store the BEST model artifacts
    best_overall_acc = 0.0
    best_fold_idx = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq_all, y_seq_all)):
        print(f"\n--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        
        # 1. SPLIT DATA
        X_train_raw_seq = X_seq_all[train_idx]
        y_train = y_seq_all[train_idx]
        
        X_val_raw_seq = X_seq_all[val_idx]
        y_val = y_seq_all[val_idx]
        
        # 2. FEATURE SELECTION (Using ONLY Train Data)
        # Flatten train sequences to shape (N_samples * Time, N_Features) for PCA
        X_train_flat = np.vstack(X_train_raw_seq)
        
        # Select Features
        print(f"  Selecting features based on training set...")
        selected_features = select_features_via_pca(X_train_flat, all_feature_names)
        print(f"  Selected {len(selected_features)} features.")
        
        # Map indices mapping
        feature_indices = [all_feature_names.index(f) for f in selected_features]
        
        # Filter Train and Val sequences to selected features
        X_train_filtered = [seq[:, feature_indices] for seq in X_train_raw_seq]
        X_val_filtered = [seq[:, feature_indices] for seq in X_val_raw_seq]
        
        # 3. PAD SEQUENCES
        X_train_padded = pad_sequences(X_train_filtered, maxlen=SEQ_LENGTH, dtype='float32')
        X_val_padded = pad_sequences(X_val_filtered, maxlen=SEQ_LENGTH, dtype='float32')
        
        l_train = torch.tensor([len(x) for x in X_train_filtered], dtype=torch.long)
        l_val = torch.tensor([len(x) for x in X_val_filtered], dtype=torch.long)
        
        # 4. DATALOADERS
        class_counts = np.bincount(y_train)
        class_weights = 1. / (class_counts + 1e-6)
        sample_weights = torch.tensor([class_weights[y] for y in y_train], dtype=torch.float32)
        weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_dataset = ExpressionDataset(list(X_train_padded), y_train, l_train)
        val_dataset = ExpressionDataset(list(X_val_padded), y_val, l_val) # No weighted sampler for Val
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=weighted_sampler, drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        # 5. MODEL SETUP
        feature_dim = len(selected_features)
        num_classes = len(unique_labels)
        model = OptimizedCNNLSTM(feature_dim, 64, num_classes, num_layers=3).to(device)
        
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        
        print(f"  Model on: {next(model.parameters()).device}")
        
        # 6. TRAINING LOOP
        epochs = 150 # Reduced from 500 for responsiveness, 500 is overkill for small data usually
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience = 20
        counter = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y, batch_lengths in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X, batch_lengths)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            
            model.eval()
            val_total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for val_X, val_y, val_lengths in val_loader:
                    val_outputs = model(val_X, val_lengths)
                    val_loss = criterion(val_outputs, val_y)
                    val_total_loss += val_loss.item()
                    val_pred = val_outputs.argmax(dim=1)
                    correct += (val_pred == val_y).sum().item()
                    total += val_y.size(0)
            
            val_loss_avg = val_total_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = correct / total if total > 0 else 0
            
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss_avg
                counter = 0
                torch.save({'model': model.state_dict(), 'features': selected_features}, f"fold_{fold}_best.pth")
            else:
                counter += 1
                if counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Fold {fold+1} Best Val Acc: {best_val_acc*100:.2f}%")
        fold_accs.append(best_val_acc)
        
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_fold_idx = fold

    print(f"\nFinal CV Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")
    
    # Save Final Best Model
    if best_fold_idx != -1:
        print(f"Saving Best Model from Fold {best_fold_idx+1} to {MODEL_FILE}")
        best_state = torch.load(f"fold_{best_fold_idx}_best.pth")
        
        # We need to save the selected FEATURES list too, otherwise we can't infer later!
        # The model structural weights depend on the exact input features.
        final_save = {
            'model_state_dict': best_state['model'],
            'selected_features': best_state['features'],
            'input_dim': len(best_state['features']),
            'num_classes': len(unique_labels),
            'label_map': label_map
        }
        torch.save(final_save, MODEL_FILE)
        
        # Save selected features to CSV for reference/inference usage
        # (Though we effectively need to use the LIST saved in the .pth for production)
        pd.DataFrame({'Feature': best_state['features']}).to_csv(SELECTED_FEATURES_CSV, index=False)

    # Cleanup
    for f in range(NUM_FOLDS):
        fname = f"fold_{f}_best.pth"
        if os.path.exists(fname): os.remove(fname)

if __name__ == "__main__":
    df = extract_data_from_videos()
    if df is not None:
        # train_model handles the selection internally now
        train_model(df)
