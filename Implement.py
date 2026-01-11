import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.fft import fft
from scipy.signal import find_peaks
import threading
import time
import json

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Landmark Indices ---
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

# --- Buffers ---
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

# --- Global ---
frame_global = None
features_global = {
    'Eye': {}, 'Brow': {}, 'Cheek': {}, 'Mouth': {}, 'Lips': {}, 'Jaw': {}
}
prev_landmarks_global = None

# --- Surface Vector Split Function ---
def compute_surface_vectors_split(landmarks, prev_landmarks, left_idx, right_idx):
    if prev_landmarks is None or landmarks is None:
        zero = {'vectors': np.array([]), 'positions': [], 'norms': np.array([]), 'mean_mag': 0.0, 'var': 0.0, 'angle': 0.0}
        return {'left': zero, 'right': zero}

    def process_side(idx_list):
        vectors, positions, norms = [], [], []
        for idx in idx_list:
            if idx >= len(landmarks) or idx >= len(prev_landmarks):
                continue
            curr = np.array(landmarks[idx])
            prev = np.array(prev_landmarks[idx])
            vec = curr - prev
            norm = np.linalg.norm(vec)
            norms.append(norm)
            vectors.append(vec / norm if norm > 1e-6 else vec)
            positions.append(curr)
        vectors = np.array(vectors) if vectors else np.array([])
        norms = np.array(norms) if norms else np.array([])

        mean_mag = np.mean(norms) if len(norms) > 0 else 0.0
        var = np.var(norms * 1000) if len(norms) > 0 else 0.0
        angle = 0.0
        if len(vectors) > 0:
            xy = vectors[:, :2]
            avg = np.mean(xy, axis=0)
            n = np.linalg.norm(avg)
            if n > 1e-6:
                avg /= n
                angle = np.arctan2(avg[1], avg[0])
        return {'vectors': vectors, 'positions': positions, 'norms': norms, 'mean_mag': mean_mag, 'var': var, 'angle': angle}

    return {'left': process_side(left_idx), 'right': process_side(right_idx)}

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


# --- CHEEK ---
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

# --- EYE ---
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

# --- JAW ---
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

# --- Video Processing ---
def process_video():
    global frame_global, features_global, prev_landmarks_global
    cap = cv2.VideoCapture(0)
    prev_landmarks = None

    region_configs = [
        ('Brow', left_brow_idx_surface, right_brow_idx_surface, (0, 255, 0)),
        ('Cheek', left_cheek_idx_surface, right_cheek_idx_surface, (255, 0, 0)),
        ('Eye', left_eye_idx_surface, right_eye_idx_surface, (0, 0, 255)),
        ('Jaw', left_jaw_idx_surface, right_jaw_idx_surface, (255, 255, 0)),
        ('Lips', left_lip_idx_surface, right_lip_idx_surface, (255, 0, 255)),
        ('Mouth', left_mouth_idx_surface, right_mouth_idx_surface, (0, 255, 255)),
    ]

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        landmarks = None

        if results.multi_face_landmarks:
            lmks = results.multi_face_landmarks[0].landmark
            landmarks = [[lm.x, lm.y, lm.z] for lm in lmks]
            h, w, _ = frame.shape
            for idx in set(brow_landmarks_idx + cheek_landmarks_idx + eye_landmarks_idx + jaw_landmarks_idx + lip_landmarks_idx + mouth_landmarks_idx):
                if idx < len(lmks):
                    lm = lmks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

            # Draw arrows
            for name, l_idx, r_idx, color in region_configs:
                data = compute_surface_vectors_split(landmarks, prev_landmarks, l_idx, r_idx)
                for side_key, side_data in [('left', data['left']), ('right', data['right'])]:
                    positions = side_data['positions']
                    mean_mag = side_data['mean_mag']
                    angle = side_data['angle']

                    if len(positions) == 0 or mean_mag < 0.0008:
                        continue

                    # Compute average direction vector from angle
                    avg_dir = np.array([np.cos(angle), np.sin(angle)])

                    # Compute center position
                    center_pos = np.mean(positions, axis=0) if positions else np.array([0, 0, 0])
                    center_x = int(center_pos[0] * w)
                    center_y = int(center_pos[1] * h)

                    # Scale arrow length
                    base_scale = 20
                    max_length = 25
                    min_length = 3
                    arrow_length = min(max(mean_mag * w * base_scale * 30, min_length), max_length)

                    dx = avg_dir[0] * arrow_length
                    dy = avg_dir[1] * arrow_length

                    end_x = int(center_x + dx)
                    end_y = int(center_y + dy)

                    # Draw the arrow
                    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 1, tipLength=0.25)

                    # Label
                    label = name[0] + ('L' if side_key == 'left' else 'R')
                    cv2.putText(frame, label, (center_x - 20, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Compute features
        features_global.update({
            'Brow': compute_brow_features(landmarks, prev_landmarks),
            'Cheek': compute_cheek_features(landmarks, prev_landmarks),
            'Eye': compute_eye_features(landmarks, prev_landmarks),
            'Jaw': compute_jaw_features(landmarks, prev_landmarks),
            'Lips': compute_lips_features(landmarks, prev_landmarks),
            'Mouth': compute_mouth_features(landmarks, prev_landmarks),
        })

        prev_landmarks = landmarks
        frame_global = frame.copy()
        time.sleep(0.03)

    cap.release()

# --- Flask App ---
from flask import Flask, Response, jsonify

app = Flask(__name__)

def gen_frames():
    while True:
        if frame_global is not None:
            ret, buf = cv2.imencode('.jpg', frame_global)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/features')
def features(): return jsonify(features_global)

@app.route('/')
def index():
    return """
    <html><head><style>
        body {background:#f8f8f8;font-family:Arial;text-align:center;}
        h1{margin-bottom:20px;}
        .buttons{margin-bottom:20px;}
        .button{padding:10px 20px;border-radius:20px;border:none;cursor:pointer;}
        .open{background:#90ee90;color:black;}
        .close{background:#ff0000;color:white;}
        .camera-space{width:800px;height:600px;background:#ddd;margin:0 auto 20px;display:none;}
        .camera-space img{width:100%;height:100%;object-fit:cover;}
        .biomarker-container{display:flex;justify-content:center;flex-wrap:wrap;}
        .biomarker-box{width:380px;margin:5px;padding:10px;background:#eee;overflow-y:auto;height:600px;}
        .biomarker-box h3{margin:0 0 10px;text-align:center;}
        h4{margin:10px 0 5px;color:#333;}
        p{margin:2px 0;font-size:0.9em;}
    </style></head><body>
        <h1>Facial Biomarker Detection</h1>
        <div class="buttons">
            <button class="button open" onclick="openCamera()">Open Camera +</button>
            <button class="button close" onclick="closeCamera()">Close Camera -</button>
        </div>
        <div class="camera-space" id="cameraSpace"><img id="video" src="/video_feed"></div>
        <div class="biomarker-container">
            <div class="biomarker-box"><h3>Eye</h3><div id="Eye"></div></div>
            <div class="biomarker-box"><h3>Brow</h3><div id="Brow"></div></div>
            <div class="biomarker-box"><h3>Cheek</h3><div id="Cheek"></div></div>
            <div class="biomarker-box"><h3>Mouth</h3><div id="Mouth"></div></div>
            <div class="biomarker-box"><h3>Lips</h3><div id="Lips"></div></div>
            <div class="biomarker-box"><h3>Jaw</h3><div id="Jaw"></div></div>
        </div>
        <script>
            let intervalId;
            function openCamera() {
                document.getElementById('cameraSpace').style.display = 'block';
                intervalId = setInterval(updateFeatures, 100);
            }
            function closeCamera() {
                document.getElementById('cameraSpace').style.display = 'none';
                clearInterval(intervalId);
                ['Eye','Brow','Cheek','Mouth','Lips','Jaw'].forEach(id=>document.getElementById(id).innerHTML='');
            }
            function updateFeatures() {
                fetch('/features').then(r=>r.json()).then(data=>{
                    for (const [key, feats] of Object.entries(data)) {
                        const box = document.getElementById(key);
                        box.innerHTML = '';
                        const left=[], right=[], other=[];
                        for (const [k,v] of Object.entries(feats)) {
                            if (k.includes('Left surface')) left.push(k.replace('Left surface ','')+': '+v.toFixed(4));
                            else if (k.includes('Right surface')) right.push(k.replace('Right surface ','')+': '+v.toFixed(4));
                            else other.push(k+': '+v.toFixed(4));
                        }
                        const add = (title, arr) => {
                            if (arr.length>0) {
                                const h=document.createElement('h4'); h.textContent=title; box.appendChild(h);
                                arr.forEach(t=>{const p=document.createElement('p');p.textContent=t;box.appendChild(p);});
                            }
                        };
                        add('Left Surface Vectors', left);
                        add('Right Surface Vectors', right);
                        add('Other Biomarkers', other);
                    }
                });
            }
        </script>
    </body></html>
    """

# --- Start ---
if __name__ == '__main__':
    t = threading.Thread(target=process_video, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000)


