import os
import cv2
import torch
import numpy as np
import pandas as pd
import mediapipe as mp

from extract_pipeline1_features import extract_features, buffers
from model_arch import AdvancedCNNLSTM

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIDEO_ROOT = "expressions"
OUTPUT_CSV = "outputs/landmark_surface_features.csv"
MODEL_PATH = "landmark_facial_expression_model.pth"

INPUT_DIM = 166
HIDDEN_DIM = 512
NUM_CLASSES = 8

# ================= MODEL =================
model = AdvancedCNNLSTM(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================= MEDIAPIPE =================
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

rows = []

# ================= PROCESS =================
for expression in os.listdir(VIDEO_ROOT):
    expr_path = os.path.join(VIDEO_ROOT, expression)
    if not os.path.isdir(expr_path):
        continue

    video_files = [v for v in os.listdir(expr_path) if v.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Processing folder '{expression}': {len(video_files)} videos found.")

    for idx, video_name in enumerate(video_files):
        video_path = os.path.join(expr_path, video_name)
        print(f"  [{idx+1}/{len(video_files)}] Processing {video_name}...", end="\r")
        cap = cv2.VideoCapture(video_path)

        for b in buffers.values():
            b.clear()

        prev_feature = None
        prev_delta = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                continue

            lm = result.multi_face_landmarks[0].landmark
            landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

            # features_166 is for the model (not used for this CSV output currently unless desired, user asked for surface stats)
            features_166, surface_stats, _ = extract_features(landmarks)
            
            # Helper to get stats safely
            def get_stat(region, side, metric):
                key = f"{region}_{side}"
                if key in surface_stats and metric in surface_stats[key]:
                    return surface_stats[key][metric]
                return 0.0

            row = [expression, video_name, frame_idx]
            
            # Order: Brow, Eye, Cheek, Mouth, Lips, Jaw (Left then Right)
            regions = ['brow', 'eye', 'cheek', 'mouth', 'lips', 'jaw']
            for reg in regions:
                for side in ['left', 'right']:
                    row.append(get_stat(reg, side, 'mean_mag'))
                    row.append(get_stat(reg, side, 'var'))
                    row.append(get_stat(reg, side, 'angle'))

            rows.append(row)
            frame_idx += 1

        cap.release()

# ================= SAVE =================
cols = ["Expression", "Video", "Frame"]
regions = ['Brow', 'Eye', 'Cheek', 'Mouth', 'Lips', 'Jaw']
for reg in regions:
    for side in ['Left', 'Right']:
        cols.append(f"{side}_{reg}_Mag")
        cols.append(f"{side}_{reg}_Var")
        cols.append(f"{side}_{reg}_Ang")

df = pd.DataFrame(rows, columns=cols)

os.makedirs("outputs", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved landmark surface features → {OUTPUT_CSV}")
