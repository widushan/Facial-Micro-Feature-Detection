
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import mediapipe as mp
import pandas as pd
from extract_pipeline1_features import extract_features, reset_buffers

# --- MODEL ARCHITECTURE (Must match Pipeline2.ipynb) ---
# --- MODEL ARCHITECTURE (Matching Pipeline2.ipynb and Checkpoint) ---
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        # Score time steps based on hidden representation (hidden_dim * 2 for Bidirectional)
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        # Generate importance weights for each frame in the sequence
        # (B, T, 2*H) -> (B, T, 1) through Linear
        scores = self.attention(lstm_output)
        weights = F.softmax(scores, dim=1)
        # Create a single context vector based on weighted features
        # (B, T, 2*H) * (B, T, 1) -> sum over T -> (B, 2*H)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

class EnhancedFACSModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout_rate=0.3):
        super(EnhancedFACSModel, self).__init__()
        
        # Initial 1D Convolution for local temporal smoothing
        # Note: Checkpoint expects keys "conv.0", "conv.1" etc.
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(2) # Reduces sequence length by 50%
        )
        
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.attention = TemporalAttention(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1) # Prep for Conv1d: (batch, features, seq_len)
        x = self.conv(x).permute(0, 2, 1) # Back to (batch, half_seq_len, 256)
        
        # Adjust lengths for MaxPool1d(2)
        # Note: In inference with batch_size=1, packing isn't critical but we keep logic consistent
        # adj_lengths = torch.clamp(lengths // 2, min=1).cpu()
        # packed = pack_padded_sequence(x, adj_lengths, batch_first=True, enforce_sorted=False)
        # lstm_out, _ = self.lstm(packed)
        # lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Simplified for inference loop (no packing needed for single batch without padding)
        lstm_out, _ = self.lstm(x)
        
        context = self.attention(lstm_out)
        return self.fc(context)

# --- CONFIG ---
EXPRESSIONS_DIR = "expressions"
MODEL_PATH = "facs_au_model_enhanced.pth" 
OUTPUT_CSV = "outputs/facs_au_features.csv" # To store extracted AUs and statistics
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # 1. Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. Setup Output CSV
    # Columns: Expression, Video, Frame, [AU1_int, AU1_pres, ...]
    au_keys = [
        'AU1_intensity', 'AU1_presence', 'AU2_intensity', 'AU2_presence', 
        'AU4_intensity', 'AU4_presence', 'AU5_intensity', 'AU5_presence',
        'AU6_intensity', 'AU6_presence', 'AU7_intensity', 'AU7_presence',
        'AU8_intensity', 'AU8_presence', 'AU9_intensity', 'AU9_presence',
        'AU10_intensity', 'AU10_presence', 'AU11_intensity', 'AU11_presence',
        'AU12_intensity', 'AU12_presence', 'AU13_intensity', 'AU13_presence',
        'AU14_intensity', 'AU14_presence', 'AU15_intensity', 'AU15_presence',
        'AU16_intensity', 'AU16_presence', 'AU17_intensity', 'AU17_presence',
        'AU18_intensity', 'AU18_presence', 'AU20_intensity', 'AU20_presence',
        'AU22_intensity', 'AU22_presence', 'AU23_intensity', 'AU23_presence',
        'AU24_intensity', 'AU24_presence', 'AU25_intensity', 'AU25_presence',
        'AU26_intensity', 'AU26_presence', 'AU27_intensity', 'AU27_presence',
        'AU28_intensity', 'AU28_presence', 'AU43_intensity', 'AU43_presence',
        'AU45_intensity', 'AU45_presence', 'AU46_intensity', 'AU46_presence'
    ]
    
    # Surface Vector Keys
    surface_keys = []
    regions = ['Brow', 'Eye', 'Cheek', 'Mouth', 'Lips', 'Jaw']
    for reg in regions:
        for side in ['L', 'R']:
            surface_keys.append(f"{side}_{reg}_Mag")
            surface_keys.append(f"{side}_{reg}_Var")
            surface_keys.append(f"{side}_{reg}_Ang")

    csv_headers = ["Expression", "Video", "Frame"] + au_keys + surface_keys
    csv_data = []

    # 3. Load Model (Sketchy part due to missing Scaler/PCA)
    # 3. Load Model
    print("Loading model architecture...")
    # Matches Pipeline2: input_dim=150 (pca), hidden=128, num_classes=8, num_layers=2
    model = EnhancedFACSModel(input_dim=150, num_classes=8, hidden_dim=128, num_layers=2).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load model state dict: {e}")
            print("Running in Feature Extraction Only mode.")
            model = None
    else:
        print(f"Warning: Model file {MODEL_PATH} not found.")
        print("Running in Feature Extraction Only mode.")
        model = None

    print("\nIMPORTANT: To run actual model inference, you need the fitted StandardScaler and PCA models from training.")
    print("If outputting AUs and Surface vectors to CSV, model loading is optional (but recommended if available).")

    # 4. Processing Loop & CSV Writing
    if not os.path.exists(EXPRESSIONS_DIR):
        print(f"Error: Directory {EXPRESSIONS_DIR} not found.")
        return
        
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    # Open CSV in append mode or write header first
    file_exists = os.path.exists(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        # Use simple string join for efficiency and control or pandas to_csv chunking
        # Here we just write header once then rows
        f.write(",".join(csv_headers) + "\n")
        
    print(f"Processing videos in {EXPRESSIONS_DIR}...")
    
    for root, dirs, files in os.walk(EXPRESSIONS_DIR):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                expression_label = os.path.basename(os.path.dirname(video_path))
                print(f"Processing {file}...", end='\r')
                
                cap = cv2.VideoCapture(video_path)
                reset_buffers()
                frame_idx = 0
                
                video_rows = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = mp_face_mesh.process(rgb)
                    
                    if results.multi_face_landmarks:
                        lm = results.multi_face_landmarks[0].landmark
                        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                        
                        # Extract!
                        flat_feats, surface_stats, aus = extract_features(landmarks)
                        
                        # Save AU data + Surface Vector Data
                        row = [expression_label, file, str(frame_idx)]
                        
                        # 1. AUs
                        for k in au_keys:
                            if k in aus:
                                row.append(str(aus[k]))
                            else:
                                row.append("0.0")

                        # 2. Surface Stats
                        for reg in regions:
                            for side in ['left', 'right']:
                                key = f"{reg.lower()}_{side}" 
                                if key in surface_stats:
                                    stats = surface_stats[key]
                                    row.append(str(stats.get('mean_mag', 0.0)))
                                    row.append(str(stats.get('var', 0.0)))
                                    row.append(str(stats.get('angle', 0.0)))
                                else:
                                    row.extend(["0.0", "0.0", "0.0"])
                        
                        video_rows.append(",".join(row))
                        
                cap.release()
                
                # Write rows for this video immediately to disk
                if video_rows:
                    with open(OUTPUT_CSV, 'a', newline='') as f:
                        f.write("\n".join(video_rows) + "\n")

    print(f"\nProcessing complete. Features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
