
import numpy as np
import collections
from collections import deque
from scipy.spatial import Delaunay
from scipy.fft import fft
from scipy.signal import find_peaks
import math

# ================= GLOBAL BUFFERS (Matching Pipeline 2) =================
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
        
        # DEBUG PRINT
        # if len(curr_pos) < 3:
        #    print(f"DEBUG: Not enough points for surface vectors. Len: {len(curr_pos)}")

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

        vectors = np.array(triangle_vectors) if triangle_vectors else np.array([])
        norms = triangle_norms
        positions = [np.mean([curr_pos[i] for i in s], axis=0) for s in tri.simplices]

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

# === FINAL UNIFIED & ENHANCED FACS ACTION UNITS DETECTION ===
def compute_action_units(landmarks, prev_landmarks, region_features):
    """
    Computes FACS Action Units (AU1 to AU66) with intensity and binary presence.
    Requires MediaPipe FaceMesh with refine_landmarks=True for iris landmarks (468-477).
    Uses normalized landmarks (relative to nose tip) and region_features from vTwo.py.
    """
    if landmarks is None or len(landmarks) < 478:  # 478+ landmarks needed for iris
        return {}

    au = {}

    # Normalize all landmarks relative to nose tip (landmark index 1)
    nose_tip = np.array(landmarks[1])
    norm_landmarks = [np.array(lm) - nose_tip for lm in landmarks]

    # ====================== FACIAL ACTION UNITS (AU1 - AU46) ======================

    # AU1: Inner Brow Raiser
    inner_brow_ys = [norm_landmarks[i][1] for i in inner_brow_idx]
    au['AU1_intensity'] = -np.mean(inner_brow_ys)
    au['AU1_presence'] = 1 if au['AU1_intensity'] > 0.02 else 0

    # AU2: Outer Brow Raiser
    left_outer_y = np.mean([norm_landmarks[i][1] for i in left_brow_idx[-3:]])
    right_outer_y = np.mean([norm_landmarks[i][1] for i in right_brow_idx[-3:]])
    au['AU2_intensity'] = -(left_outer_y + right_outer_y) / 2
    au['AU2_presence'] = 1 if au['AU2_intensity'] > 0.02 else 0

    # AU4: Brow Lowerer
    au['AU4_intensity'] = region_features['Brow'].get('Brow velocity (mean)', 0) if au['AU1_intensity'] < -0.01 else 0
    au['AU4_presence'] = 1 if au['AU4_intensity'] > 0.01 else 0

    # AU5: Upper Lid Raiser
    au['AU5_intensity'] = region_features['Eye'].get('Eye ratio (mean)', 0)
    au['AU5_presence'] = 1 if au['AU5_intensity'] > 0.15 else 0

    # AU6: Cheek Raiser
    au['AU6_intensity'] = region_features['Cheek'].get('Cheek raise (mean)', 0)
    au['AU6_presence'] = 1 if au['AU6_intensity'] > 0.03 else 0

    # AU7: Lid Tightener (squint)
    au['AU7_intensity'] = region_features['Eye'].get('Eye squint velocity (mean)', 0)
    au['AU7_presence'] = 1 if au['AU7_intensity'] > 0.01 else 0

    # AU8: Lips Toward Each Other
    lip_open = np.linalg.norm(norm_landmarks[13] - norm_landmarks[14])
    au['AU8_intensity'] = max(0.05 - lip_open, 0)
    au['AU8_presence'] = 1 if au['AU8_intensity'] > 0.02 else 0

    # AU9: Nose Wrinkler
    upper_lip_y = norm_landmarks[13][1]
    au['AU9_intensity'] = -upper_lip_y
    au['AU9_presence'] = 1 if au['AU9_intensity'] > 0.03 else 0

    # AU10: Upper Lip Raiser
    au['AU10_intensity'] = region_features['Lips'].get('Lip velocity (mean)', 0) if upper_lip_y < 0 else 0
    au['AU10_presence'] = 1 if au['AU10_intensity'] > 0.01 else 0

    # AU11: Nasolabial Deepener
    au['AU11_intensity'] = (
        region_features['Cheek'].get('Left surface variance mean', 0) +
        region_features['Cheek'].get('Right surface variance mean', 0)
    )
    au['AU11_presence'] = 1 if au['AU11_intensity'] > 0.02 else 0

    # AU12: Lip Corner Puller (smile)
    left_corner = norm_landmarks[61][1] - norm_landmarks[17][1]
    right_corner = norm_landmarks[291][1] - norm_landmarks[17][1]
    au['AU12_intensity'] = (left_corner + right_corner) / 2 if left_corner > 0 and right_corner > 0 else 0
    au['AU12_presence'] = 1 if au['AU12_intensity'] > 0.02 else 0

    # AU13: Cheek Puffer
    au['AU13_intensity'] = region_features['Cheek'].get('Cheek puff micro-expression variance mean', 0)
    au['AU13_presence'] = 1 if au['AU13_intensity'] > 0.02 else 0

    # AU14: Dimpler
    au['AU14_intensity'] = region_features['Cheek'].get('Cheek surface variance (current)', 0)
    au['AU14_presence'] = 1 if au['AU14_intensity'] > 0.015 else 0

    # AU15: Lip Corner Depressor (frown)
    au['AU15_intensity'] = -(left_corner + right_corner) / 2 if left_corner < 0 and right_corner < 0 else 0
    au['AU15_presence'] = 1 if au['AU15_intensity'] > 0.02 else 0

    # AU16: Lower Lip Depressor
    lower_lip_y = norm_landmarks[14][1]
    au['AU16_intensity'] = lower_lip_y
    au['AU16_presence'] = 1 if au['AU16_intensity'] > 0.03 else 0

    # AU17: Chin Raiser
    chin_y = norm_landmarks[152][1]
    au['AU17_intensity'] = -chin_y
    au['AU17_presence'] = 1 if au['AU17_intensity'] > 0.02 else 0

    # AU18: Lip Pucker
    lip_width = np.linalg.norm(norm_landmarks[61] - norm_landmarks[291])
    au['AU18_intensity'] = max(0.1 - lip_width, 0)
    au['AU18_presence'] = 1 if au['AU18_intensity'] > 0.02 else 0

    # AU20: Lip Stretcher
    au['AU20_intensity'] = lip_width
    au['AU20_presence'] = 1 if au['AU20_intensity'] > 0.15 else 0

    # AU22: Lip Funneler
    au['AU22_intensity'] = abs(region_features['Mouth'].get('Left surface dominant angle mean', 0)) + \
                            abs(region_features['Mouth'].get('Right surface dominant angle mean', 0))
    au['AU22_presence'] = 1 if au['AU22_intensity'] > 0.5 else 0

    # AU23: Lip Tightener
    au['AU23_intensity'] = region_features['Lips'].get('Lip micro-expression variance mean', 0)
    au['AU23_presence'] = 1 if au['AU23_intensity'] < 0.01 else 0

    # AU24: Lip Pressor
    au['AU24_intensity'] = region_features['Lips'].get('Lip velocity (mean)', 0) if lip_open < 0.02 else 0
    au['AU24_presence'] = 1 if au['AU24_intensity'] > 0.01 else 0

    # AU25: Lips Part
    au['AU25_intensity'] = region_features['Lips'].get('Lip opening (mean)', 0)
    au['AU25_presence'] = 1 if au['AU25_intensity'] > 0.05 else 0

    # AU26: Jaw Drop
    au['AU26_intensity'] = region_features['Jaw'].get('Jaw opening (mean)', 0)
    au['AU26_presence'] = 1 if au['AU26_intensity'] > 0.1 else 0

    # AU27: Mouth Stretch
    mouth_width = np.linalg.norm(norm_landmarks[61] - norm_landmarks[291])
    au['AU27_intensity'] = au['AU26_intensity'] * mouth_width
    au['AU27_presence'] = 1 if au['AU27_intensity'] > 0.1 else 0

    # AU28: Lip Suck
    au['AU28_intensity'] = region_features['Lips'].get('Left surface variance mean', 0) + \
                           region_features['Lips'].get('Right surface variance mean', 0)
    au['AU28_presence'] = 1 if au['AU28_intensity'] > 0.05 else 0

    # AU43: Eyes Closed
    au['AU43_intensity'] = max(0.2 - region_features['Eye'].get('Eye ratio (mean)', 0), 0)
    au['AU43_presence'] = 1 if au['AU43_intensity'] > 0.1 else 0

    # AU45: Blink
    au['AU45_intensity'] = region_features['Eye'].get('Blink rate', 0)
    au['AU45_presence'] = 1 if au['AU45_intensity'] > 0.5 else 0

    # AU46: Wink
    au['AU46_intensity'] = abs(region_features['Eye'].get('Eye ratio (mean)', 0) - \
                               region_features['Eye'].get('Eye ratio (std)', 0))
    au['AU46_presence'] = 1 if au['AU46_intensity'] > 0.05 else 0

    return au

# Global internal state
_prev_landmarks = None

def extract_features(landmarks):
    global _prev_landmarks
    
    # Run the modular feature extractors
    brow = compute_brow_features(landmarks, _prev_landmarks)
    cheek = compute_cheek_features(landmarks, _prev_landmarks)
    eye = compute_eye_features(landmarks, _prev_landmarks)
    jaw = compute_jaw_features(landmarks, _prev_landmarks)
    lips = compute_lips_features(landmarks, _prev_landmarks)
    mouth = compute_mouth_features(landmarks, _prev_landmarks)
    
    region_features = {
        'Brow': brow,
        'Cheek': cheek,
        'Eye': eye,
        'Jaw': jaw,
        'Lips': lips,
        'Mouth': mouth
    }
    
    # Compute AUs
    aus = compute_action_units(landmarks, _prev_landmarks, region_features)

    # --- Construct Surface Stats for Web App ---
    # App expects keys: brow_left, brow_right, eye_left, etc.
    # containing: mean_mag, var, angle
    # SCALING: Multiply by 1000 (mag) and 1e6 (var) to make micro-movements visible in UI
    surface_stats = {}
    
    # Map region names to their global index variables
    surface_indices = {
        'brow': (left_brow_idx_surface, right_brow_idx_surface),
        'cheek': (left_cheek_idx_surface, right_cheek_idx_surface),
        'eye': (left_eye_idx_surface, right_eye_idx_surface),
        'jaw': (left_jaw_idx_surface, right_jaw_idx_surface),
        'lips': (left_lip_idx_surface, right_lip_idx_surface),
        'mouth': (left_mouth_idx_surface, right_mouth_idx_surface)
    }

    for region_name, (l_idx, r_idx) in surface_indices.items():
        # Compute surface vectors directly
        # Note: landmarks are normalized [0,1]. Differences are very small (e.g. 1e-4).
        s = compute_surface_vectors_split(landmarks, _prev_landmarks, l_idx, r_idx)
        
        # Scale for display suitability
        l_stat = s['left'].copy()
        r_stat = s['right'].copy()
        
        l_stat['mean_mag'] *= 1000.0
        l_stat['var'] *= 1000000.0
        
        r_stat['mean_mag'] *= 1000.0
        r_stat['var'] *= 1000000.0
        
        surface_stats[f"{region_name}_left"] = l_stat
        surface_stats[f"{region_name}_right"] = r_stat

    # Debug print for user verification
    if 'brow_left' in surface_stats:
        print(f"DEBUG: Brow Left Mag (Scaled): {surface_stats['brow_left']['mean_mag']:.4f}")

    # Update prev landmarks
    _prev_landmarks = landmarks
    
    # Flatten features for model input (166 dim)
    # Order MUST match Pipeline1.ipynb: Brow, Cheek, Eye, Jaw, Lips, Mouth
    flat = []
    for region in ['Brow', 'Cheek', 'Eye', 'Jaw', 'Lips', 'Mouth']:
        if region in region_features:
            for v in region_features[region].values():
                flat.append(float(v))
                
    # Fallback if empty (e.g. first frame)
    if not flat:
        flat = [0.0] * 166
        
    return np.array(flat, dtype=np.float32), surface_stats, aus

def reset_buffers():
    global _prev_landmarks
    _prev_landmarks = None
    for b in [brow_raise_buffer, brow_left_raise_buffer, brow_right_raise_buffer, brow_inner_raise_buffer, brow_vel_buffer, brow_surface_var_buffer, brow_surface_dir_buffer,
              cheek_raise_buffer, cheek_vel_buffer, cheek_surface_var_buffer, cheek_surface_dir_buffer,
              eye_ratio_buffer, eye_vel_buffer, blink_buffer, eye_surface_var_buffer, eye_surface_dir_buffer,
              jaw_open_buffer, jaw_vel_buffer, jaw_surface_var_buffer, jaw_surface_dir_buffer,
              lips_open_buffer, lips_vel_buffer, lips_surface_var_buffer, lips_surface_dir_buffer,
              mouth_open_buffer, mouth_vel_buffer, mouth_surface_var_buffer, mouth_surface_dir_buffer]:
        b.clear()
