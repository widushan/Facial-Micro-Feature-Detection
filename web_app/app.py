
import os
import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Add parent directory to path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from extract_pipeline1_features import extract_features, reset_buffers
    from model_arch import AdvancedCNNLSTM
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure you are running this from the right directory or that {parent_dir} contains the required files.")

app = Flask(__name__)

# --- Config ---
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(parent_dir, "landmark_facial_expression_model.pth")
SCALER_PATH = os.path.join(parent_dir, "scaler.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 166
HIDDEN_DIM = 512
NUM_CLASSES = 8
CLASSES = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# --- Load Model ---
model = AdvancedCNNLSTM(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
else:
    print(f"❌ Model file not found at {MODEL_PATH}")

# --- Load Scaler ---
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
else:
    print(f"⚠️ Scaler not found at {SCALER_PATH}. Predictions might be inaccurate (likely 'Fearful').")


# --- Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(save_path)

    # Dictionary to hold time-series data for the response
    # Structure: {'brow_left': {'mean_mag': [], ...}, ...}
    series_data = {}
    
    # Initialize keys based on expected structure
    regions = ['brow', 'eye', 'cheek', 'mouth', 'lips', 'jaw']
    sides = ['left', 'right']
    metrics = ['mean_mag', 'var', 'angle']
    
    for r in regions:
        for s in sides:
            key = f"{r}_{s}"
            series_data[key] = {m: [] for m in metrics}

    # Model Input Sequence
    feature_seq = []

    # Process
    try:
        cap = cv2.VideoCapture(save_path)
        
        # Initialize FaceMesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            # Reset buffers (Critical for new video)
            reset_buffers()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks_obj = results.multi_face_landmarks[0].landmark
                    # Convert to list [x, y, z]
                    landmarks = [[lm.x, lm.y, lm.z] for lm in landmarks_obj]
                    
                    # Extract
                    # features_166 is np.array of shape (166,)
                    # surface_stats is dict {'brow_left': {...}, ...}
                    feat_166, surf_stats, _ = extract_features(landmarks)
                    
                    feature_seq.append(feat_166)
                    
                    # Accumulate Surface Stats
                    for key in series_data:
                        if key in surf_stats:
                            for metric in metrics:
                                val = surf_stats[key].get(metric, 0.0)
                                if isinstance(val, (np.floating, float)):
                                    series_data[key][metric].append(float(val))
                                else:
                                    series_data[key][metric].append(0.0)
                        else:
                            # Fill with 0 if missing
                            for metric in metrics:
                                series_data[key][metric].append(0.0)
                
                else:
                    # No face detected
                    # We might want to skip or append zeros. 
                    # For charts, probably skip to avoid noise or append previous?
                    # Let's skip model input but pad charts with 0?
                    # For simplicity, just continue
                    continue
        
        cap.release()
        
        # --- Inference ---
        emotion_prediction = "--"
        if len(feature_seq) > 0:
            # Convert to numpy
            X = np.array(feature_seq)
            
            # Normalize sequence if scaler is available
            if scaler:
                try:
                    X = scaler.transform(X)
                except Exception as e:
                    print(f"Normalization failed: {e}")

            # Prepare Input: (1, T, 166)
            input_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            # Lengths: tensor([T])
            lengths = torch.tensor([len(feature_seq)], dtype=torch.long).to("cpu")
            
            with torch.no_grad():
                # Model forward expects (x, lengths)
                output = model(input_tensor, lengths)
                # Output shape (1, 8)
                probs = torch.softmax(output, dim=1)
                predicted_idx = torch.argmax(probs, dim=1).item()
                emotion_prediction = CLASSES[predicted_idx]
                
                print(f"Prediction: {emotion_prediction} (Index: {predicted_idx})")
                print(f"Probabilities: {probs.cpu().numpy()}")

        # Clean up upload
        try:
            os.remove(save_path)
        except:
            pass
            
        return jsonify({
            'emotion': emotion_prediction,
            'surface_stats': series_data
        })

    except Exception as e:
        print(f"Processing Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible if needed, default port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
