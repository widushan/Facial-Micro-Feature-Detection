import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from math import atan2, degrees
import mediapipe as mp
from scipy.spatial import Delaunay

# ==============================
# CONFIG & PATHS
# ==============================
EXPRESSIONS_DIR = "expressions"
OUTPUT_DIR = "outputs"
LANDMARK_MODEL_PATH = "landmark_facial_expression_model.pth"
FACS_MODEL_PATH = "facs_au_model_enhanced.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# FACIAL REGION INDICES (12 Vectors: 6 Regions x L/R)
# Derived from Pipeline indices
# ==============================
REGIONS = {
    "left_brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "left_cheek": [205, 206, 216, 204, 207, 114, 115, 116],
    "right_cheek": [425, 426, 436, 424, 427, 343, 344, 345],
    "left_eye": [33, 160, 158, 133, 153, 144, 145, 159],
    "right_eye": [362, 385, 387, 263, 373, 374, 380, 386],
    "left_jaw": [152, 176, 136, 172],
    "right_jaw": [397, 365, 366, 379, 400, 378, 377],
    "left_lip": [61, 78, 80, 81, 82, 84, 91, 95],
    "right_lip": [291, 308, 310, 311, 312, 314, 321, 324],
    "left_mouth": [61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 17],
    "right_mouth": [291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375]
}

# ==============================
# UTILS & FEATURE EXTRACTION
# ==============================
def normalize_landmarks(landmarks):
    """Normalize landmarks by distance from nose tip."""
    nose_tip = np.array(landmarks[1])
    normalized = []
    for lm in landmarks:
        lm_arr = np.array(lm)
        dist = np.linalg.norm(lm_arr - nose_tip) + 1e-6
        normalized.append((lm_arr - nose_tip) / dist)
    return np.array(normalized)

def compute_surface_stats(landmarks, prev_landmarks, idx_list):
    """Calculates magnitude, variance, and angle using triangulation."""
    if prev_landmarks is None or len(idx_list) < 3:
        return 0.0, 0.0, 0.0

    curr_pos = landmarks[idx_list]
    prev_pos = prev_landmarks[idx_list]
    points2d = curr_pos[:, :2]

    try:
        tri = Delaunay(points2d)
    except:
        return 0.0, 0.0, 0.0

    norms = []
    vectors = []
    for simplex in tri.simplices:
        v = (curr_pos[simplex] - prev_pos[simplex]).mean(axis=0)
        norm = np.linalg.norm(v)
        norms.append(norm)
        vectors.append(v)

    mean_mag = np.mean(norms) if norms else 0.0
    variance = np.var(norms) if norms else 0.0
    avg_v = np.mean(vectors, axis=0) if vectors else [0, 0]
    angle = degrees(atan2(avg_v[1], avg_v[0]))

    return mean_mag, variance, angle

# ==============================
# VIDEO PROCESSING ENGINE
# ==============================
def process_all_videos(model, model_name):
    """Iterates through expressions and builds one master feature list."""
    master_data = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    for expression in os.listdir(EXPRESSIONS_DIR):
        expr_path = os.path.join(EXPRESSIONS_DIR, expression)
        if not os.path.isdir(expr_path): continue

        for video_file in os.listdir(expr_path):
            if not video_file.lower().endswith((".mp4", ".avi", ".mov")): continue
            
            video_path = os.path.join(expr_path, video_file)
            cap = cv2.VideoCapture(video_path)
            prev_landmarks = None
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0].landmark
                    curr_landmarks = np.array([[p.x, p.y, p.z] for p in lm])
                    
                    if prev_landmarks is not None:
                        row = {
                            'video_name': video_file,
                            'expression': expression,
                            'frame': frame_count
                        }
                        # Extract 12 Surface Vectors (36 values total per frame)
                        for region_name, idxs in REGIONS.items():
                            mag, var, ang = compute_surface_stats(curr_landmarks, prev_landmarks, idxs)
                            row[f"{region_name}_magnitude"] = mag
                            row[f"{region_name}_variance"] = var
                            row[f"{region_name}_angle"] = ang
                        
                        master_data.append(row)

                    prev_landmarks = curr_landmarks
                    frame_count += 1
            cap.release()
            print(f"Processed {video_file} for {model_name}")

    return pd.DataFrame(master_data)

# ==============================
# EXECUTION
# ==============================
if __name__ == "__main__":
    # Note: Using torch.load directly as you provided. 
    # Ensure classes are defined if models were saved as state_dicts.
    print("Loading models...")
    landmark_model = torch.load(LANDMARK_MODEL_PATH, map_location=DEVICE)
    facs_model = torch.load(FACS_MODEL_PATH, map_location=DEVICE)

    # Generate Landmark Master CSV
    print("\nStarting Landmark Model Feature Extraction...")
    df_landmark = process_all_videos(landmark_model, "Landmark Model")
    df_landmark.to_csv(os.path.join(OUTPUT_DIR, "landmark_master_features.csv"), index=False)

    # Generate FACS Master CSV
    print("\nStarting FACS Model Feature Extraction...")
    df_facs = process_all_videos(facs_model, "FACS Model")
    df_facs.to_csv(os.path.join(OUTPUT_DIR, "facs_master_features.csv"), index=False)

    print(f"\n✅ Extraction Complete. Files saved in {OUTPUT_DIR}")