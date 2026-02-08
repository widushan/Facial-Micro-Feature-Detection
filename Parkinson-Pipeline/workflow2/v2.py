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
from scipy.signal import get_window

# Options
DATASET_DIR = "../../PD Videos/Training"
OUTPUT_DIR = "."
OUTPUT_CSV_HEALTHY = "left_eye_magnitude_healthy.csv"
OUTPUT_CSV_PARKINSON = "left_eye_magnitude_parkinson.csv"

# --- Landmark Indices ---
# Left Eye Surface Indices from v1.py
left_eye_idx_surface = [33, 160, 158, 133, 153, 144, 145, 159]

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
        
        # EMA Filter
        smoothed = self.alpha * current + (1 - self.alpha) * self.prev_smoothed
        self.prev_smoothed = smoothed
        
        return smoothed.tolist()

def compute_surface_magnitude(landmarks, prev_landmarks, idx_list):
    if prev_landmarks is None or landmarks is None or not idx_list:
        return 0.0

    curr_pos = []
    prev_pos = []
    for idx in idx_list:
        if idx >= len(landmarks) or idx >= len(prev_landmarks):
            continue
        curr_pos.append(np.array(landmarks[idx]))
        prev_pos.append(np.array(prev_landmarks[idx]))
    
    if len(curr_pos) < 3:
        return 0.0

    points2d = np.array([p[:2] for p in curr_pos])

    try:
        tri = Delaunay(points2d)
    except:
        return 0.0

    triangle_norms = []

    for simplex in tri.simplices:
        i1, i2, i3 = simplex
        v1 = curr_pos[i1] - prev_pos[i1]
        v2 = curr_pos[i2] - prev_pos[i2]
        v3 = curr_pos[i3] - prev_pos[i3]
        mean_v = (v1 + v2 + v3) / 3
        norm = np.linalg.norm(mean_v)
        triangle_norms.append(norm)

    triangle_norms = np.array(triangle_norms)
    mean_mag = np.mean(triangle_norms) if len(triangle_norms) > 0 else 0.0
    
    return mean_mag

def extract_features():
    global _prev_landmarks_global
    print("Starting Feature Extraction for Left Eye Magnitude...")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return None, None

    # Separate lists for Healthy and Parkinson
    data_healthy = []
    data_parkinson = []
    
    # Assuming classes are 'Healthy' and 'Parkinson' (or 'PD') based on folders
    # Reuse directory detection from v1.py
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
                        
                        # 1. Normalize
                        lm_list_norm = normalize_with_rotation(lm_list, None)
                        
                        # 2. Smooth
                        lm_list_smooth = smoother.update(lm_list_norm)
                        
                        # Extract Left Eye Magnitude
                        mag = compute_surface_magnitude(lm_list_smooth, _prev_landmarks_global, left_eye_idx_surface)
                        
                        features = {
                            'Video': video_name,
                            'Label': category,
                            'Frame': frame_count,
                            'Left_Eye_Mag': mag
                        }
                        
                        if "Healthy" in category or "healthy" in category:
                             data_healthy.append(features)
                        else:
                             data_parkinson.append(features)
                        
                        _prev_landmarks_global = lm_list_smooth
            cap.release()

    df_healthy = pd.DataFrame(data_healthy)
    df_parkinson = pd.DataFrame(data_parkinson)
    
    if not df_healthy.empty:
        df_healthy.to_csv(OUTPUT_CSV_HEALTHY, index=False)
        print(f"Saved Healthy data to {OUTPUT_CSV_HEALTHY} ({len(df_healthy)} frames)")
        
    if not df_parkinson.empty:
        df_parkinson.to_csv(OUTPUT_CSV_PARKINSON, index=False)
        print(f"Saved Parkinson data to {OUTPUT_CSV_PARKINSON} ({len(df_parkinson)} frames)")
        
    return df_healthy, df_parkinson

# --- Frequency Analysis ---
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

def analyze_and_plot(df_healthy, df_parkinson):
    print("\nStarting Analysis & Visualization...")
    
    # Combine for some plots
    df_all = pd.concat([df_healthy, df_parkinson], ignore_index=True)
    if df_all.empty:
        print("No data to analyze.")
        return

    # 1. Time Series Plot (Raw Signal)
    plt.figure(figsize=(12, 6))
    
    # Plot a few samples from each
    def plot_samples(df, label, color):
        videos = df['Video'].unique()[:3] # Top 3 videos
        for video in videos:
            video_df = df[df['Video'] == video].sort_values('Frame')
            plt.plot(video_df['Left_Eye_Mag'].values, color=color, alpha=0.5, label=label)
            
    if not df_healthy.empty: plot_samples(df_healthy, 'Healthy', 'green')
    if not df_parkinson.empty: plot_samples(df_parkinson, 'Parkinson', 'red')
    
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title("Left Eye Magnitude Signal (Time Domain)")
    plt.xlabel("Frame")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig("left_eye_time_series.png")
    print("Saved left_eye_time_series.png")

    # 2. FFT Spectrum
    SAMPLE_RATE = 30.0
    FFT_SIZE = 256
    freq_axis = scipy.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
    
    fft_spectra = {'Healthy': [], 'Parkinson': []}
    
    def process_fft(df, label):
        videos = df['Video'].unique()
        for video in videos:
            video_df = df[df['Video'] == video].sort_values('Frame')
            signal = video_df['Left_Eye_Mag'].values
            
            # Normalize and Window
            signal = signal - np.mean(signal)
            if len(signal) < FFT_SIZE:
                signal = np.pad(signal, (0, FFT_SIZE - len(signal)), 'constant')
            else:
                signal = signal[:FFT_SIZE]
            
            windowed = signal * get_window("hann", len(signal))
            spectrum = scipy.fft.fft(windowed, n=FFT_SIZE)
            magnitude = np.abs(spectrum[:int(FFT_SIZE/2) + 1])
            fft_spectra[label].append(magnitude)

    if not df_healthy.empty: process_fft(df_healthy, 'Healthy')
    if not df_parkinson.empty: process_fft(df_parkinson, 'Parkinson')
    
    plt.figure(figsize=(10, 6))
    for label, spectra in fft_spectra.items():
        if spectra:
            avg_spec = np.mean(spectra, axis=0)
            color = 'green' if label == 'Healthy' else 'red'
            plt.plot(freq_axis, avg_spec, label=label, color=color)
            
    plt.title("Average FFT Spectrum - Left Eye Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("left_eye_fft_spectrum.png")
    print("Saved left_eye_fft_spectrum.png")

    # 3. Mel Spectrograms
    NUM_MEL_FILTERS = 12
    mel_filters = compute_mel_filterbank(NUM_MEL_FILTERS, FFT_SIZE, SAMPLE_RATE, 0, SAMPLE_RATE/2)
    
    def create_mel_heatmap(spectra_list, label):
        if not spectra_list: return
        
        # Calculate Mel Energy for each video
        mel_energies_list = []
        for mag_spec in spectra_list:
            power_spec = (mag_spec ** 2) / FFT_SIZE
            mel_energy = np.dot(mel_filters, power_spec)
            log_mel = 10 * np.log10(mel_energy + 1e-9)
            mel_energies_list.append(log_mel)
            
        avg_mel_vector = np.mean(mel_energies_list, axis=0)
        
        plt.figure(figsize=(8, 4))
        # Reshape for heatmap (1 x Bands)
        sns.heatmap(avg_mel_vector.reshape(1, -1), cmap='viridis', annot=True, 
                    xticklabels=[f'Band {i+1}' for i in range(NUM_MEL_FILTERS)], yticklabels=[label])
        plt.title(f"Average Mel-Spectrogram Vector - {label}")
        plt.tight_layout()
        filename = f"left_eye_mel_spectrogram_{label.lower()}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")

    create_mel_heatmap(fft_spectra['Healthy'], 'Healthy')
    create_mel_heatmap(fft_spectra['Parkinson'], 'Parkinson')

if __name__ == "__main__":
    # Check if CSVs exist to avoid re-running slow extraction
    if os.path.exists(OUTPUT_CSV_HEALTHY) and os.path.exists(OUTPUT_CSV_PARKINSON):
        print("Loading existing CSVs...")
        df_h = pd.read_csv(OUTPUT_CSV_HEALTHY)
        df_p = pd.read_csv(OUTPUT_CSV_PARKINSON)
    else:
        df_h, df_p = extract_features()
        
    analyze_and_plot(df_h, df_p)
    print("Workflow v2 Complete.")
