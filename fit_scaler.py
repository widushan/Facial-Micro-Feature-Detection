import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
import random
from sklearn.preprocessing import StandardScaler
from extract_pipeline1_features import extract_features, reset_buffers

# Config
VIDEO_ROOT = "expressions"
SCALER_PATH = "scaler.pkl"
MAX_FRAMES_FOR_FIT = 5000  # Enough samples for robust mean/std estimation

def main():
    print("Initializing MediaPipe...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    collected_features = []
    
    # 1. Find videos
    videos = []
    for root, _, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                videos.append(os.path.join(root, f))
    
    if not videos:
        print(f"No videos found in {VIDEO_ROOT}")
        return

    # Shuffle to get diverse samples (different emotions/subjects)
    random.shuffle(videos)
    print(f"Found {len(videos)} videos. Collecting {MAX_FRAMES_FOR_FIT} frames...")

    frame_count = 0
    
    for vid_idx, vid_path in enumerate(videos):
        cap = cv2.VideoCapture(vid_path)
        reset_buffers() # crucial for temporal features
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert color space
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Extract features
                landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]
                
                # We only need the flat 166-dim vector
                feat_166, _, _ = extract_features(landmarks)
                
                # Check for validity (avoid all zeros if extraction failed)
                if np.any(feat_166):
                    collected_features.append(feat_166)
                    frame_count += 1
            
            if frame_count >= MAX_FRAMES_FOR_FIT:
                break
        
        cap.release()
        print(f"  Processed {vid_idx+1}/{len(videos)}: {vid_path} ({frame_count} frames)")
        
        if frame_count >= MAX_FRAMES_FOR_FIT:
            break

    # 2. Fit Scaler
    if not collected_features:
        print("Error: No valid features collected.")
        return

    X = np.array(collected_features)
    print(f"\nFitting StandardScaler on shape {X.shape}...")
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    # 3. Save
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Mean (first 5): {scaler.mean_[:5]}")
    print(f"Scale (first 5): {scaler.scale_[:5]}")

if __name__ == "__main__":
    main()
