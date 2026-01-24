import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import sys
import os

# Import local module 'pd' to reuse code and model architecture
# This must be in the same directory as pd.py
try:
    import pd as pd_module
except ImportError:
    print("Error: Could not import 'pd.py'. Make sure this script is in the 'Parkinson-Pipeline' directory.")
    sys.exit(1)

# --- CONFIGURATION ---
MODEL_PATH = "pd_detection_model.pth"
CLASSES = ['Healthy', 'Parkinson']
SEQ_LENGTH = 120
SELECTED_FEATURES = [
    'Brow micro-expression variance mean','Brow micro-expression rapid changes count','Brow velocity (std)','Right brow raise (mean)',
    'Left brow raise (mean)','Left brow raise (std)','Inner brow raise (mean)','Inner brow raise (std)','Brow asymmetry (mean)',
    'Temporal brow asymmetry variance','Brow frequency mean','Brow peak frequency','Brow Left surface vector magnitude mean',
    'Brow Left surface variance (current)','Brow Left surface variance std','Brow Left surface variance min','Brow Left surface variance max',
    'Brow Left surface dominant angle mean','Brow Left surface dominant angle std','Brow Right surface vector magnitude mean',
    'Brow Right surface variance (current)','Brow Right surface variance mean','Brow Right surface variance min','Brow Right surface dominant angle mean',
    'Brow Right surface dominant angle std','Cheek puff micro-expression variance mean','Cheek puff rapid changes count','Cheek raise (mean)',
    'Cheek velocity (mean)','Cheek velocity (std)','Cheek frequency mean','Cheek asymmetry (mean)','Cheek Left surface vector magnitude mean',
    'Cheek Left surface variance (current)','Cheek Left surface variance mean','Cheek Left surface variance std','Cheek Left surface variance min',
    'Cheek Left surface variance max','Cheek Left surface dominant angle mean','Cheek Left surface dominant angle std','Cheek Right surface vector magnitude mean',
    'Cheek Right surface variance (current)','Cheek Right surface variance mean','Cheek Right surface variance min','Cheek Right surface variance max',
    'Cheek Right surface dominant angle mean','Cheek Right surface dominant angle std','Eye widening rapid changes count','Eye ratio (mean)',
    'Eye ratio (std)','Blink rate','Eye squint velocity (mean)','Eye squint velocity (std)','Eye Left surface vector magnitude mean',
    'Eye Left surface variance (current)','Eye Left surface variance std','Eye Left surface variance min','Eye Left surface variance max',
    'Eye Left surface dominant angle mean','Eye Right surface variance (current)','Eye Right surface variance mean','Eye Right surface variance std',
    'Eye Right surface variance min','Eye Right surface variance max','Eye Right surface dominant angle mean','Jaw opening (mean)','Jaw opening (std)',
    'Jaw opening (min)','Jaw velocity (mean)','Jaw velocity (std)','Jaw asymmetry (mean)','Jaw asymmetry (max)','Jaw rapid changes count',
    'Jaw frequency mean','Jaw peak frequency','Jaw Left surface variance (current)','Jaw Left surface variance min','Jaw Left surface dominant angle mean',
    'Jaw Left surface dominant angle std','Jaw Right surface vector magnitude mean','Jaw Right surface variance (current)','Jaw Right surface variance std',
    'Jaw Right surface variance min','Jaw Right surface variance max','Jaw Right surface dominant angle mean','Jaw Right surface dominant angle std',
    'Lip micro-expression rapid changes count','Lip opening (mean)','Lip opening (min)','Lip opening (max)','Lip velocity (std)',
    'Lip significant movements count','Lip frequency mean','Lip peak frequency','Lip corner asymmetry (mean)','Lip corner asymmetry (max)',
    'Lip Left surface vector magnitude mean','Lip Left surface variance (current)','Lip Left surface variance mean','Lip Left surface variance std',
    'Lip Left surface variance min','Lip Left surface variance max','Lip Left surface dominant angle mean','Lip Left surface dominant angle std',
    'Lip Right surface variance (current)','Lip Right surface variance min','Lip Right surface dominant angle mean','Lip Right surface dominant angle std',
    'Mouth micro-expression rapid changes count','Mouth opening (mean)','Mouth opening (min)','Mouth opening (max)','Mouth velocity (std)',
    'Mouth significant movements count','Mouth frequency mean','Mouth peak frequency','Mouth corner asymmetry (mean)','Mouth corner asymmetry (max)',
    'Mouth Left surface vector magnitude mean','Mouth Left surface variance (current)','Mouth Left surface variance mean','Mouth Left surface variance min',
    'Mouth Left surface variance max','Mouth Left surface dominant angle mean','Mouth Left surface dominant angle std','Mouth Right surface vector magnitude mean',
    'Mouth Right surface variance (current)','Mouth Right surface variance std','Mouth Right surface variance min','Mouth Right surface dominant angle mean',
    'Mouth Right surface dominant angle std'
]

# --- SYSTEM CLASS ---
class ParkinsonDetectionSystem:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = len(SELECTED_FEATURES)
        self.hidden_dim = 128
        self.num_classes = len(CLASSES)
        
        # Initialize Model using the class from pd_module to ensure compatibility
        try:
            self.model = pd_module.OptimizedCNNLSTM(self.feature_dim, self.hidden_dim, self.num_classes, num_layers=3)
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Region configs for visualization (reused concept from vTwo.py/pd.py)
        # We need access to the index lists. pd_module has them.
        self.region_configs = [
            ('Brow', pd_module.left_brow_idx_surface, pd_module.right_brow_idx_surface, (0, 255, 0)),
            ('Cheek', pd_module.left_cheek_idx_surface, pd_module.right_cheek_idx_surface, (255, 0, 0)),
            ('Eye', pd_module.left_eye_idx_surface, pd_module.right_eye_idx_surface, (0, 0, 255)),
            ('Jaw', pd_module.left_jaw_idx_surface, pd_module.right_jaw_idx_surface, (255, 255, 0)),
            ('Lips', pd_module.left_lip_idx_surface, pd_module.right_lip_idx_surface, (255, 0, 255)),
            ('Mouth', pd_module.left_mouth_idx_surface, pd_module.right_mouth_idx_surface, (0, 255, 255)),
        ]

    def process_video_file(self, video_path):
        if not os.path.exists(video_path):
            print(f"Error processing video: File not found {video_path}")
            return

        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Reset buffers in the pd module to clear previous state
        pd_module.reset_buffers()
        pd_module._prev_landmarks_global = None # Ensure global state is clear
        
        extracted_data = []
        frame_idx = 0
        
        # For visualization purposes, we might want to scale the video if it's too large
        
        with self.mp_face_mesh.FaceMesh(
            refine_landmarks=True, max_num_faces=1, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Convert for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = face_mesh.process(image_rgb)
                image_rgb.flags.writeable = True
                
                # Visualization frame
                vis_frame = frame.copy()
                h, w, _ = vis_frame.shape

                landmarks_list = None
                if results.multi_face_landmarks:
                    # Only take the first face
                    landmarks = results.multi_face_landmarks[0].landmark
                    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                    
                    # Draw Mesh
                    # mp.solutions.drawing_utils.draw_landmarks(vis_frame, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

                    # Use global var from pd_module to track state
                    # We pass _prev_landmarks_global from the module, but we iterate it ourselves here
                    # Actually pd_module functions don't take prev_landmarks as argument, wait.
                    # Looking at pd.py:
                    # def compute_brow_features(landmarks, prev_landmarks):
                    # It DOES take prev_landmarks.
                    # But extract_data_from_videos in pd.py manages _prev_landmarks_global.
                    
                    prev_lm = pd_module._prev_landmarks_global
                    
                    # Normalize first
                    lm_list_norm = pd_module.normalize_for_rotation_distance(landmarks_list, prev_lm)
                    
                    # Extract Features using pd_module functions
                    features_dict = {}
                    features_dict.update(pd_module.compute_brow_features(lm_list_norm, prev_lm))
                    features_dict.update(pd_module.compute_cheek_features(lm_list_norm, prev_lm))
                    features_dict.update(pd_module.compute_eye_features(lm_list_norm, prev_lm))
                    features_dict.update(pd_module.compute_jaw_features(lm_list_norm, prev_lm))
                    features_dict.update(pd_module.compute_lips_features(lm_list_norm, prev_lm))
                    features_dict.update(pd_module.compute_mouth_features(lm_list_norm, prev_lm))
                    
                    features_dict['Frame'] = frame_idx
                    extracted_data.append(features_dict)
                    
                    # IMPORTANT: Update the global state in pd_module so next iteration uses it
                    pd_module._prev_landmarks_global = lm_list_norm
                    
                    # --- Visualization of Surface Vectors ---
                    # We can reuse the logic from vTwo.py/pd.py or write a simplified version
                    # Use the raw landmarks for visualization on screen (not normalized)
                    # But compute_surface_vectors_split calculates vectors based on movement
                    # Let's just draw the arrows if we have movement. 
                    # Note: We computed surface features above using normalized landmarks.
                    # To visualize on the original frame, we probably want to visualize the *result* or just the landmarks.
                    # Let's draw landmarks for now to show tracking.
                    
                    # Draw regions
                    # Added cheek, jaw, and mouth landmarks as requested
                    for idx_set in [
                        pd_module.left_eye_idx, pd_module.right_eye_idx, 
                        pd_module.left_brow_idx, pd_module.right_brow_idx, 
                        pd_module.lip_landmarks_idx,
                        pd_module.left_cheek_idx, pd_module.right_cheek_idx,
                        pd_module.jaw_landmarks_idx,
                        pd_module.mouth_landmarks_idx
                    ]:
                        for idx in idx_set:
                            if idx < len(landmarks):
                                lm = landmarks[idx]
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(vis_frame, (cx, cy), 1, (0, 255, 0), -1)

                
                # Show Feature Extraction Progress
                cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, "Extracting Features...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Feature Extraction', vis_frame)
                if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not extracted_data:
            print("No face detected or no data extracted.")
            return

        print("\nFeature Extraction Complete.")
        
        # --- PREPARE DATA FOR INFERENCE ---
        df = pd.DataFrame(extracted_data)
        
        # Fill NaNs
        df = df.fillna(0)
        
        # Filter Columns
        # Ensure all expected columns exist
        missing_cols = [c for c in SELECTED_FEATURES if c not in df.columns]
        if missing_cols:
            print(f"Warning: {len(missing_cols)} expected features are missing. Filling with 0.")
            for c in missing_cols:
                df[c] = 0.0
                
        # Reorder and select
        X_df = df[SELECTED_FEATURES]
        
        # Save CSV
        output_csv = "video_extracted_features.csv"
        X_df.to_csv(output_csv, index=False)
        print(f"Frame-by-frame features saved to {output_csv}")
        
        # Prepare Sequence
        X = X_df.values
        seq_len = len(X)
        
        # Pad/Truncate to SEQ_LENGTH (120)
        # Using the logic from pd.py: pad_sequences uses 'post' padding by default in the function definition 
        # but usage in train_model is: X_padded = pad_sequences(X_seq, maxlen=SEQ_LENGTH, dtype='float32')
        # Let's use the pd_module.pad_sequences if available, else do it manually.
        # pad_sequences is defined in pd.py
        
        # Create a list of sequences (just 1 here)
        X_seq = [X[:SEQ_LENGTH] if seq_len > SEQ_LENGTH else X]
        
        # Pad
        X_padded = pd_module.pad_sequences(X_seq, maxlen=SEQ_LENGTH, dtype='float32')
        
        # Convert to Tensor
        X_tensor = torch.tensor(X_padded, dtype=torch.float32).to(self.device)
        # Length tensor
        real_len = min(seq_len, SEQ_LENGTH)
        L_tensor = torch.tensor([real_len], dtype=torch.long).to(self.device)
        
        # Inference
        print("Running Inference...")
        with torch.no_grad():
            outputs = self.model(X_tensor, L_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = CLASSES[pred_idx]
            confidence = probs[0][pred_idx].item()
            
        print("\n" + "="*40)
        print(f"PREDICTION: {pred_label.upper()}")
        print(f"Confidence: {confidence:.2f}")
        print("="*40 + "\n")


if __name__ == "__main__":
    import argparse
    
    # We can accept an argument or ask interactively
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Please provide the video path.")
        video_path = input("Video Path: ").strip()
        # Remove quotes if user added them
        video_path = video_path.replace('"', '').replace("'", "")
    
    # Check if path exists, if not try ../
    if not os.path.exists(video_path):
        parent_path = os.path.join("..", video_path)
        if os.path.exists(parent_path):
            print(f"Video found in parent directory: {parent_path}")
            video_path = parent_path
        
    system = ParkinsonDetectionSystem(MODEL_PATH)
    system.process_video_file(video_path)
