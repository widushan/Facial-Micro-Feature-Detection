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
                        
                        lm_list_norm = normalize_for_rotation_distance(lm_list, _prev_landmarks_global)
                        
                        # Extract Surface Vectors
                        brow = compute_surface_vectors_split(lm_list_norm, _prev_landmarks_global, left_brow_idx_surface, right_brow_idx_surface)
                        cheek = compute_surface_vectors_split(lm_list_norm, _prev_landmarks_global, left_cheek_idx_surface, right_cheek_idx_surface)
                        eye = compute_surface_vectors_split(lm_list_norm, _prev_landmarks_global, left_eye_idx_surface, right_eye_idx_surface)
                        jaw = compute_surface_vectors_split(lm_list_norm, _prev_landmarks_global, left_jaw_idx_surface, right_jaw_idx_surface)
                        lips = compute_surface_vectors_split(lm_list_norm, _prev_landmarks_global, left_lip_idx_surface, right_lip_idx_surface)
                        mouth = compute_surface_vectors_split(lm_list_norm, _prev_landmarks_global, left_mouth_idx_surface, right_mouth_idx_surface)
                        
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
                        
                        _prev_landmarks_global = lm_list_norm
            cap.release()

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

if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing data {OUTPUT_CSV}. Loading...")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = extract_features()
    
    if df is not None:
        analyze_features(df)
