import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.fft
from scipy.signal import get_window, spectrogram

# --- Options ---
DATASET_DIR = "../../PD Videos/Training"
ROOT_OUTPUT_DIR = "Feature_Analysis"
DURATION_LIMIT = 5.0  # seconds

# --- Landmark Indices ---
landmark_groups = {
    'Brow_Left': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'Brow_Right': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    'Cheek_Left': [205, 206, 216, 204, 207, 114, 115, 116],
    'Cheek_Right': [425, 426, 436, 424, 427, 343, 344, 345],
    'Eye_Left': [33, 160, 158, 133, 153, 144, 145, 159],
    'Eye_Right': [362, 385, 387, 263, 373, 374, 380, 386],
    'Jaw_Left': [152, 176, 136, 172],
    'Jaw_Right': [397, 365, 366, 379, 400, 378, 377],
    'Lips_Left': [61, 78, 80, 81, 82, 84, 91, 95],
    'Lips_Right': [291, 308, 310, 311, 312, 314, 321, 324],
    'Mouth_Left': [61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 17],
    'Mouth_Right': [291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375]
}

_prev_landmarks_global = None

def reset_buffers():
    global _prev_landmarks_global
    _prev_landmarks_global = None

# --- Improved Normalization (Rotation + Translation + Scale) ---
def normalize_with_rotation(landmarks):
    if landmarks is None or len(landmarks) == 0:
        return landmarks
    
    # 1. Translation: Center on nose tip (idx 1)
    nose_tip = np.array(landmarks[1])
    centered = [np.array(lm) - nose_tip for lm in landmarks]
    
    # 2. Rotation: Align eyes horizontally
    left_eye = centered[33]
    right_eye = centered[263]
    delta = right_eye - left_eye 
    angle = np.arctan2(delta[1], delta[0])
    
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    
    rotated = []
    for lm in centered:
        v2d = np.dot(R, lm[:2])
        rotated.append(np.array([v2d[0], v2d[1], lm[2]]))
        
    # 3. Scale: Inter-ocular distance (IOD)
    iod = np.linalg.norm(np.array(rotated[263]) - np.array(rotated[33])) + 1e-6
    normalized = [lm / iod for lm in rotated]
    
    return normalized

# --- Smoothing Class ---
class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_smoothed = None
        
    def update(self, current_landmarks):
        if current_landmarks is None:
            return None
        current = np.array(current_landmarks)
        if self.prev_smoothed is None:
            self.prev_smoothed = current
            return current.tolist()
        smoothed = self.alpha * current + (1 - self.alpha) * self.prev_smoothed
        self.prev_smoothed = smoothed
        return smoothed.tolist()

def compute_surface_vectors(landmarks, prev_landmarks, idx_list):
    if prev_landmarks is None or landmarks is None or not idx_list:
        return 0.0, 0.0

    curr_pos = []
    prev_pos = []
    for idx in idx_list:
        if idx >= len(landmarks) or idx >= len(prev_landmarks):
            continue
        curr_pos.append(np.array(landmarks[idx]))
        prev_pos.append(np.array(prev_landmarks[idx]))
    
    if len(curr_pos) < 3:
        return 0.0, 0.0

    points2d = np.array([p[:2] for p in curr_pos])

    try:
        tri = Delaunay(points2d)
    except:
        return 0.0, 0.0

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
            
        v_a = points2d[i2] - points2d[i1]
        v_b = points2d[i3] - points2d[i1]
        cross_prod = v_a[0] * v_b[1] - v_a[1] * v_b[0]
        area = 0.5 * np.abs(cross_prod)
        triangle_areas.append(area)

    mean_mag = np.mean(triangle_norms) if len(triangle_norms) > 0 else 0.0
    
    angle = 0.0
    if triangle_vectors and triangle_areas:
        weighted_vectors = np.array(triangle_vectors) * np.array(triangle_areas)[:, np.newaxis]
        total_area = np.sum(triangle_areas)
        if total_area > 0:
            avg = np.sum(weighted_vectors[:, :2], axis=0) / total_area
            n = np.linalg.norm(avg)
            if n > 1e-6:
                angle = np.arctan2(avg[1], avg[0])
    
    return mean_mag, angle

def extract_all_features():
    global _prev_landmarks_global
    print(f"Starting Multi-Feature Extraction with {DURATION_LIMIT}s limit...")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return None

    all_data = [] # List of dicts
    
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    mp_face_mesh = mp.solutions.face_mesh
    
    for category in classes:
        print(f"Processing category: {category}")
        video_files = glob.glob(os.path.join(DATASET_DIR, category, "*.mp4"))
        
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            
            # --- Duration Filtering ---
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or np.isnan(fps):
                fps = 30.0 
            
            duration = total_frames / fps
            
            if duration < DURATION_LIMIT:
                print(f"Skipping {video_name}: Duration too short ({duration:.2f}s < {DURATION_LIMIT}s)")
                cap.release()
                continue
            
            max_frames = int(DURATION_LIMIT * fps)
            print(f"Processing {video_name}: Duration {duration:.2f}s, extracting first {DURATION_LIMIT}s ({max_frames} frames)")
            
            reset_buffers()
            smoother = LandmarkSmoother(alpha=0.6)
            
            with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                frame_count = 0
                while cap.isOpened() and frame_count < max_frames:
                    success, image = cap.read()
                    if not success: break
                    frame_count += 1
                    
                    image.flags.writeable = False
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)
                    
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                        
                        lm_list_norm = normalize_with_rotation(lm_list)
                        lm_list_smooth = smoother.update(lm_list_norm)
                        
                        features = {
                            'Video': video_name,
                            'Label': category,
                            'Frame': frame_count,
                            'FPS': fps
                        }
                        
                        # Extract all 24 features
                        for group_name, indices in landmark_groups.items():
                            mag, ang = compute_surface_vectors(lm_list_smooth, _prev_landmarks_global, indices)
                            features[f"{group_name}_Mag"] = mag
                            features[f"{group_name}_Angle"] = ang
                        
                        all_data.append(features)
                        _prev_landmarks_global = lm_list_smooth
            cap.release()

    if not all_data:
        print("No features extracted.")
        return None
        
    return pd.DataFrame(all_data)

def hz_to_mel(frequencies):
    return 2595 * np.log10(1 + frequencies / 700.0)

def mel_to_hz(mels):
    return 700 * (10**(mels / 2595.0) - 1)

def compute_mel_filterbank(num_filters, fft_size, sample_rate, low_freq, high_freq):
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((num_filters, int(fft_size / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return filters

def process_and_organize_outputs(df):
    if df is None or df.empty:
        return

    print("\nOrganizing outputs and generating spectrograms...")
    
    # Get list of all feature columns (suffixes _Mag and _Angle)
    feature_cols = [c for c in df.columns if c.endswith("_Mag") or c.endswith("_Angle")]
    
    # Parameters for spectrogram
    SAMPLE_RATE = 30.0  
    NPERSEG = 32        
    NOVERLAP = 16       
    NFFT = 128          
    NUM_MEL_FILTERS = 20 

    mel_filters = compute_mel_filterbank(NUM_MEL_FILTERS, NFFT, SAMPLE_RATE, 0, SAMPLE_RATE/2)

    for feature in feature_cols:
        feature_dir = os.path.join(ROOT_OUTPUT_DIR, feature)
        os.makedirs(feature_dir, exist_ok=True)
        
        # Save CSV for this feature
        feature_df = df[['Video', 'Label', 'Frame', feature, 'FPS']]
        csv_path = os.path.join(feature_dir, "records.csv")
        feature_df.to_csv(csv_path, index=False)
        
        # Create Spectrogram Subdirs
        spec_root = os.path.join(feature_dir, "Spectrograms")
        os.makedirs(os.path.join(spec_root, "Healthy"), exist_ok=True)
        os.makedirs(os.path.join(spec_root, "Parkinson"), exist_ok=True)
        
        # Generate Spectrograms for each video
        unique_videos = feature_df['Video'].unique()
        for video in unique_videos:
            v_df = feature_df[feature_df['Video'] == video].sort_values('Frame')
            signal = v_df[feature].values
            label = v_df['Label'].iloc[0]
            
            if len(signal) < NPERSEG:
                continue
            
            signal = signal - np.mean(signal)
            
            f, t, Zxx = spectrogram(signal, fs=SAMPLE_RATE, window='hann', 
                                   nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, 
                                   detrend=False, mode='magnitude')
            
            Zxx = (Zxx ** 2) / NFFT
            
            if mel_filters.shape[1] != Zxx.shape[0]:
                 temp_filters = compute_mel_filterbank(NUM_MEL_FILTERS, (Zxx.shape[0]-1)*2, SAMPLE_RATE, 0, SAMPLE_RATE/2)
                 mel_spec = np.dot(temp_filters, Zxx)
            else:
                 mel_spec = np.dot(mel_filters, Zxx)
            
            mel_spec_db = 10 * np.log10(mel_spec + 1e-9)
            
            # Create figure without background
            fig = plt.figure(figsize=(6, 3), frameon=False)

            # Remove margins
            plt.axes([0, 0, 1, 1])
            plt.axis('off')

            # Plot ONLY spectrogram
            plt.imshow(
                mel_spec_db,
                aspect='auto',
                origin='lower',
                cmap='magma'
            )

            subdir = "Healthy" if "Healthy" in label or "healthy" in label else "Parkinson"
            output_filename = f"{os.path.splitext(video)[0]}_mel.png"
            output_path = os.path.join(spec_root, subdir, output_filename)

            # Save without padding or background
            plt.savefig(
                output_path,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True
            )
            plt.close(fig)
            
        print(f"Finished processing feature: {feature}")

if __name__ == "__main__":
    df = extract_all_features()
    process_and_organize_outputs(df)
    print("\nWorkflow v4 Complete.")
