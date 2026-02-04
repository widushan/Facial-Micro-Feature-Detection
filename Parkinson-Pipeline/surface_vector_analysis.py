import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.fft
from scipy.signal import get_window

# Options
DATASET_DIR = "../PD Videos/Training"
OUTPUT_DIR = "."
OUTPUT_CSV = "surface_features_raw.csv"

# --- Landmark Indices (Copied from no_augment_pd.py) ---
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

# --- Improved Normalization (Rotation + Translation + Scale) ---
def normalize_with_rotation(landmarks, prev_landmarks):
    if landmarks is None or len(landmarks) == 0:
        return landmarks
    
    # 1. Translation: Center on nose tip (idx 1)
    nose_tip = np.array(landmarks[1])
    centered = [np.array(lm) - nose_tip for lm in landmarks]
    
    # 2. Rotation: Align eyes horizontally
    # Left Eye Outer: 33, Right Eye Outer: 263
    left_eye = centered[33]
    right_eye = centered[263]
    
    # Angle of the vector connecting eyes
    delta = right_eye - left_eye 
    angle = np.arctan2(delta[1], delta[0])
    
    # Rotation Matrix (2D) to make angle 0 (horizontal)
    # We want to rotate by -angle
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    
    rotated = []
    for lm in centered:
        # Rotate x,y only
        v2d = np.dot(R, lm[:2])
        # Reconstruct (x, y, z) - z is unchanged for 2D rotation
        rotated.append(np.array([v2d[0], v2d[1], lm[2]]))
        
    # 3. Scale: Distance between eyes (or nose distance)
    # Using eye distance for scale is often more robust than nose distance
    # because nose length varies less with expression but eye distance is bone-fixed?
    # Actually, let's stick to the previous method's scale metric (distance from nose) 
    # OR better: Inter-ocular distance (IOD) which is constant for a person.
    
    # New scale: Distance between outer eye corners
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
        
        # EMA Filter: S_t = alpha * Y_t + (1-alpha) * S_{t-1}
        # alpha higher = follows input closer (less smooth, faster response)
        # alpha lower = relies more on history (more smooth, assumes landmarks are stable)
        # For micro-expressions, we want to kill jitter but keep movement. 0.6 is a good start.
        smoothed = self.alpha * current + (1 - self.alpha) * self.prev_smoothed
        self.prev_smoothed = smoothed
        
        return smoothed.tolist()

def compute_surface_vectors_split(landmarks, prev_landmarks, left_idx, right_idx):
    if prev_landmarks is None or landmarks is None:
        zero = {'mean_mag': 0.0, 'angle': 0.0}
        return {'left': zero, 'right': zero}

    def process_side(idx_list):
        if not idx_list:
            return {'mean_mag': 0.0, 'angle': 0.0}

        curr_pos = []
        prev_pos = []
        for idx in idx_list:
            if idx >= len(landmarks) or idx >= len(prev_landmarks):
                continue
            curr_pos.append(np.array(landmarks[idx]))
            prev_pos.append(np.array(prev_landmarks[idx]))
        
        if len(curr_pos) < 3:
            return {'mean_mag': 0.0, 'angle': 0.0}

        points2d = np.array([p[:2] for p in curr_pos])

        try:
            tri = Delaunay(points2d)
        except:
            return {'mean_mag': 0.0, 'angle': 0.0}

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
        
        angle = 0.0
        if triangle_vectors and triangle_areas:
            weighted_vectors = np.array(triangle_vectors) * np.array(triangle_areas)[:, np.newaxis]
            avg = np.sum(weighted_vectors[:, :2], axis=0) / np.sum(triangle_areas)
            n = np.linalg.norm(avg)
            if n > 1e-6:
                angle = np.arctan2(avg[1], avg[0])

        return {'mean_mag': mean_mag, 'angle': angle}

    return {'left': process_side(left_idx), 'right': process_side(right_idx)}

def extract_features():
    global _prev_landmarks_global
    print("Starting Surface Vector Feature Extraction...")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return None

    data = []
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    mp_face_mesh = mp.solutions.face_mesh
    
    for category in classes:
        print(f"Processing category: {category}")
        video_files = glob.glob(os.path.join(DATASET_DIR, category, "*.mp4"))
        
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            reset_buffers()
            
            # Initialize Smoother for this video
            smoother = LandmarkSmoother(alpha=0.6)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps == 0 or np.isnan(fps):
                fps = 30.0 # Default fallback
            
            with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                frame_count = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success: break
                    frame_count += 1
                    
                    image.flags.writeable = False
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)
                    
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                        
                        # 1. Normalize (Rotation + Scale)
                        lm_list_norm = normalize_with_rotation(lm_list, None)
                        
                        # 2. Smooth
                        lm_list_smooth = smoother.update(lm_list_norm)
                        
                        # Extract Surface Vectors
                        brow = compute_surface_vectors_split(lm_list_smooth, _prev_landmarks_global, left_brow_idx_surface, right_brow_idx_surface)
                        cheek = compute_surface_vectors_split(lm_list_smooth, _prev_landmarks_global, left_cheek_idx_surface, right_cheek_idx_surface)
                        eye = compute_surface_vectors_split(lm_list_smooth, _prev_landmarks_global, left_eye_idx_surface, right_eye_idx_surface)
                        jaw = compute_surface_vectors_split(lm_list_smooth, _prev_landmarks_global, left_jaw_idx_surface, right_jaw_idx_surface)
                        lips = compute_surface_vectors_split(lm_list_smooth, _prev_landmarks_global, left_lip_idx_surface, right_lip_idx_surface)
                        mouth = compute_surface_vectors_split(lm_list_smooth, _prev_landmarks_global, left_mouth_idx_surface, right_mouth_idx_surface)
                        
                        features = {
                            'Video': video_name,
                            'Label': category,
                            'Frame': frame_count,
                            'FPS': fps,
                            
                            # Brow
                            'Brow_Left_Mag': brow['left']['mean_mag'],
                            'Brow_Left_Angle': brow['left']['angle'],
                            'Brow_Right_Mag': brow['right']['mean_mag'],
                            'Brow_Right_Angle': brow['right']['angle'],
                            
                            # Cheek
                            'Cheek_Left_Mag': cheek['left']['mean_mag'],
                            'Cheek_Left_Angle': cheek['left']['angle'],
                            'Cheek_Right_Mag': cheek['right']['mean_mag'],
                            'Cheek_Right_Angle': cheek['right']['angle'],
                            
                            # Eye
                            'Eye_Left_Mag': eye['left']['mean_mag'],
                            'Eye_Left_Angle': eye['left']['angle'],
                            'Eye_Right_Mag': eye['right']['mean_mag'],
                            'Eye_Right_Angle': eye['right']['angle'],
                            
                            # Jaw
                            'Jaw_Left_Mag': jaw['left']['mean_mag'],
                            'Jaw_Left_Angle': jaw['left']['angle'],
                            'Jaw_Right_Mag': jaw['right']['mean_mag'],
                            'Jaw_Right_Angle': jaw['right']['angle'],
                            
                            # Lips
                            'Lips_Left_Mag': lips['left']['mean_mag'],
                            'Lips_Left_Angle': lips['left']['angle'],
                            'Lips_Right_Mag': lips['right']['mean_mag'],
                            'Lips_Right_Angle': lips['right']['angle'],
                            
                            # Mouth
                            'Mouth_Left_Mag': mouth['left']['mean_mag'],
                            'Mouth_Left_Angle': mouth['left']['angle'],
                            'Mouth_Right_Mag': mouth['right']['mean_mag'],
                            'Mouth_Right_Angle': mouth['right']['angle'],
                        }
                        
                        data.append(features)
                        
                        _prev_landmarks_global = lm_list_smooth
            cap.release()

    if not data:
        print("No features extracted. Check if videos exist in the dataset directory.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Extraction Complete. Saved {len(df)} frames to {OUTPUT_CSV}")
    return df

def analyze_features(df):
    print("\nStarting Feature Importance Analysis using Random Forest...")
    print("Why this method? Random Forest Feature Importance ranks features by how much they improve classification accuracy.")
    print("It effectively captures non-linear relationships and interactions between features.")
    
    # Prepare Data
    feature_cols = [c for c in df.columns if c not in ['Video', 'Label', 'Frame', 'FPS']]
    X = df[feature_cols]
    y = df['Label']
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # 5-Fold Cross Validation for robust importance estimation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    importances = np.zeros(len(feature_cols))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        clf.fit(X_train, y_train)
        importances += clf.feature_importances_
        
    importances /= 5  # Average over folds
    
    # Associate importance with feature names
    feature_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\n========================================================")
    print("TOP SURFACE VECTOR FEATURES FOR PARKINSON'S DETECTION")
    print("========================================================")
    print(feature_imp)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
    plt.title('Feature Importance for Parkinson\'s Detection (12 Surface Vectors)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('surface_vector_importance.png')
    print("\nFeature importance plot saved to 'surface_vector_importance.png'")
    
    # Detailed Report
    with open("surface_vector_analysis_report.txt", "w") as f:
        f.write("Surface Vector Feature Analysis Report\n")
        f.write("======================================\n\n")
        f.write("Method: Random Forest Feature Importance\n")
        f.write("Why: Identifies which surface vectors (Magnitude/Angle) most strongly differentiate PD vs Healthy.\n\n")
        f.write("Ranked Features:\n")
        f.write(feature_imp.to_string())
        
    return feature_imp.iloc[0]['Feature']

# --- Frequency Domain Analysis (FFT -> Mel) ---
def hz_to_mel(frequencies):
    return 2595 * np.log10(1 + frequencies / 700.0)

def mel_to_hz(mels):
    return 700 * (10**(mels / 2595.0) - 1)

def compute_mel_filterbank(num_filters, fft_size, sample_rate, low_freq, high_freq):
    """
    Compute a Mel-filterbank matrix.
    """
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

def analyze_frequency_domain(df, top_feature):
    print(f"\nStarting Frequency Domain Analysis on Top Feature: {top_feature}")
    print("Step 1: Extract Time Series -> Step 2: FFT -> Step 3: Mel Filterbank")
    
    # Parameters
    SAMPLE_RATE = 30.0  # Approx FPS
    FFT_SIZE = 256      # Next power of 2 from typical frame counts (approx 100-300 frames)
    NUM_MEL_FILTERS = 12
    
    # Groups
    processed_data = []
    
    videos = df['Video'].unique()
    
    mel_list = []
    labels = []
    
    mel_filters = compute_mel_filterbank(NUM_MEL_FILTERS, FFT_SIZE, SAMPLE_RATE, 0, SAMPLE_RATE / 2)
    
    for video in videos:
        video_df = df[df['Video'] == video].sort_values('Frame')
        signal = video_df[top_feature].values
        label = video_df['Label'].iloc[0]
        
        # Normalize signal (remove DC offset)
        signal = signal - np.mean(signal)
        
        # Windowing (Hanning)
        if len(signal) < FFT_SIZE:
             # Pad with zeros
            pad_len = FFT_SIZE - len(signal)
            signal = np.pad(signal, (0, pad_len), 'constant')
        else:
             # Truncate
             signal = signal[:FFT_SIZE]
             
        windowed_signal = signal * get_window("hann", len(signal))
        
        # FFT
        spectrum = scipy.fft.fft(windowed_signal, n=FFT_SIZE)
        magnitude_spectrum = np.abs(spectrum[:int(FFT_SIZE/2) + 1])
        
        # Power Spectrum
        power_spectrum = (magnitude_spectrum ** 2) / FFT_SIZE
        
        # Mel Filterbank Application
        mel_energies = np.dot(mel_filters, power_spectrum)
        
        # Log Mel Energies
        # Add small epsilon to avoid log(0)
        log_mel_energies = 10 * np.log10(mel_energies + 1e-9)
        
        features = {'Video': video, 'Label': label}
        for i in range(NUM_MEL_FILTERS):
            features[f'Mel_Band_{i+1}'] = log_mel_energies[i]
            
        mel_list.append(features)
        
    mel_df = pd.DataFrame(mel_list)
    output_mel_csv = "mel_frequency_features.csv"
    mel_df.to_csv(output_mel_csv, index=False)
    print(f"Mel-frequency features saved to {output_mel_csv}")
    
    # Train Classifier on Mel Features
    print("\nTraining Classifier on Mel-Frequency Features...")
    X_mel = mel_df[[c for c in mel_df.columns if 'Mel_Band' in c]]
    y_mel = mel_df['Label']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_all = []
    y_true_all = []
    
    for train_index, test_index in skf.split(X_mel, y_mel):
        X_train, X_test = X_mel.iloc[train_index], X_mel.iloc[test_index]
        y_train, y_test = y_mel.iloc[train_index], y_mel.iloc[test_index]
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        y_pred_all.extend(y_pred)
        y_true_all.extend(y_test)
        
    print("\nClassification Report (Mel-Frequency Features):")
    print(classification_report(y_true_all, y_pred_all))
    
    # Plot Mel Spectrogram (heatmap of avg mel vectors per class)
    plt.figure(figsize=(10, 6))
    avg_mel = mel_df.groupby('Label')[[c for c in mel_df.columns if 'Mel_Band' in c]].mean()
    sns.heatmap(avg_mel.T, cmap='viridis', annot=True)
    plt.title(f'Average Random Mel-Vectorgram for Top Feature: {top_feature}')
    plt.xlabel('Class')
    plt.ylabel('Mel Frequency Band')
    plt.tight_layout()
    plt.savefig('mel_vectorgram_avg.png')
    print("Average Mel-Vectorgram saved to 'mel_vectorgram_avg.png'")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing data {OUTPUT_CSV}. Loading...")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = extract_features()
    
    top_feature = None
    if df is not None:
        top_feature = analyze_features(df)
        
    if top_feature:
        analyze_frequency_domain(df, top_feature)
