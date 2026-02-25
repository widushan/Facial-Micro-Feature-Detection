import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import time

# ===============================
# 1. Configuration
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODEL_PATH = "OOF_Model_Comparison/best_efficientnet_model.pth"
TEMP_SPECTROGRAM = "temp_inference_spectrogram.png"

# Normalized Eye Indices from v3.py / v2.py
left_eye_idx_surface = [33, 160, 158, 133, 153, 144, 145, 159]

# ===============================
# 2. Model Loader
# ===============================
def load_best_model(model_path):
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using uninitialized weights.")
        
    model.to(DEVICE)
    model.eval()
    return model

# ===============================
# 3. Feature Extraction Logic
# ===============================
class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_smoothed = None
        
    def update(self, current_landmarks):
        if current_landmarks is None: return None
        current = np.array(current_landmarks)
        if self.prev_smoothed is None:
            self.prev_smoothed = current
            return current.tolist()
        smoothed = self.alpha * current + (1 - self.alpha) * self.prev_smoothed
        self.prev_smoothed = smoothed
        return smoothed.tolist()

def normalize_with_rotation(landmarks):
    if landmarks is None or len(landmarks) == 0: return landmarks
    nose_tip = np.array(landmarks[1])
    centered = [np.array(lm) - nose_tip for lm in landmarks]
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
    iod = np.linalg.norm(np.array(rotated[263]) - np.array(rotated[33])) + 1e-6
    normalized = [lm / iod for lm in rotated]
    return normalized

def compute_surface_magnitude(landmarks, prev_landmarks, idx_list):
    if prev_landmarks is None or landmarks is None or not idx_list: return 0.0
    curr_pos = [np.array(landmarks[idx]) for idx in idx_list]
    prev_pos = [np.array(prev_landmarks[idx]) for idx in idx_list]
    if len(curr_pos) < 3: return 0.0
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
        triangle_norms.append(np.linalg.norm(mean_v))
    return np.mean(triangle_norms) if triangle_norms else 0.0

def process_video_for_magnitude(video_path):
    print(f"Processing Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    mp_face_mesh = mp.solutions.face_mesh
    smoother = LandmarkSmoother(alpha=0.6)
    magnitudes = []
    prev_landmarks = None
    
    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]
                normalized = normalize_with_rotation(landmarks)
                smoothed = smoother.update(normalized)
                if prev_landmarks is not None:
                    mag = compute_surface_magnitude(smoothed, prev_landmarks, left_eye_idx_surface)
                    magnitudes.append(mag)
                prev_landmarks = smoothed
    cap.release()
    return np.array(magnitudes), fps

# ===============================
# 4. Spectrogram Generation
# ===============================
def hz_to_mel(frequencies): return 2595 * np.log10(1 + frequencies / 700.0)
def mel_to_hz(mels): return 700 * (10**(mels / 2595.0) - 1)

def compute_mel_filterbank(num_filters, fft_size, sample_rate):
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((num_filters, int(fft_size / 2 + 1)))
    for m in range(1, num_filters + 1):
        for k in range(bin_points[m-1], bin_points[m]):
            filters[m-1, k] = (k - bin_points[m-1]) / (bin_points[m] - bin_points[m-1])
        for k in range(bin_points[m], bin_points[m+1]):
            filters[m-1, k] = (bin_points[m+1] - k) / (bin_points[m+1] - bin_points[m])
    return filters

def generate_spectrogram_image(signal, sample_rate, output_path, video_name):
    # Parameters from v3.py
    NPERSEG, NOVERLAP, NFFT, NUM_MEL_FILTERS = 64, 32, 256, 40
    
    signal = signal - np.mean(signal)
    
    # Pad signal if too short
    if len(signal) < NPERSEG:
        signal = np.pad(signal, (0, NPERSEG - len(signal)), 'constant')
        
    f, t, Zxx = spectrogram(signal, fs=sample_rate, window='hann', 
                           nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend=False, mode='magnitude')
    Zxx = (Zxx ** 2) / NFFT
    mel_filters = compute_mel_filterbank(NUM_MEL_FILTERS, NFFT, sample_rate)
    mel_spec = np.dot(mel_filters, Zxx)
    mel_spec_db = 10 * np.log10(mel_spec + 1e-9)
    
    # Plotting code MUST match v3.py exactly for domain consistency
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma',
               extent=[t.min(), t.max(), 0, NUM_MEL_FILTERS])
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {video_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Filter Index')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ===============================
# 5. Inference Pipeline
# ===============================
def run_inference(input_path, model):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    results = []
    
    # Path Robustness: Check relative and common parent levels
    search_paths = [
        input_path,
        os.path.join("..", input_path),
        os.path.join("..", "..", input_path)
    ]
    
    actual_path = None
    for p in search_paths:
        if os.path.exists(p):
            actual_path = p
            break
            
    if not actual_path:
        print(f"Error: Input path '{input_path}' not found.")
        return

    if os.path.isfile(actual_path):
        video_files = [os.path.basename(actual_path)]
        video_dir = os.path.dirname(actual_path)
    else:
        video_files = [f for f in os.listdir(actual_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        video_dir = actual_path
    
    if not video_files:
        print(f"No video files found in {actual_path}")
        return

    for video_file in video_files:
        path = os.path.join(video_dir, video_file)
        mag_signal, fps = process_video_for_magnitude(path)
        
        if len(mag_signal) < 64:
            print(f"Skipping {video_file}: Too short")
            continue
            
        generate_spectrogram_image(mag_signal, fps, TEMP_SPECTROGRAM, video_file)
        
        img = Image.open(TEMP_SPECTROGRAM).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(outputs, dim=1).item()
            confidence = probs[0][pred_idx].item()
            
        label = "Parkinson" if pred_idx == 1 else "Healthy"
        print(f"Prediction for {video_file}: {label} ({confidence:.2%})")
        results.append({'Video': video_file, 'Prediction': label, 'Confidence': confidence})
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--dir", type=str, required=True, help="Path to video file or directory")
    args = parser.parse_args()
    
    model = load_best_model(MODEL_PATH)
    results_df = run_inference(args.input, model)
    
    if results_df is not None:
        print("\nFinal Results:")
        print(results_df)
