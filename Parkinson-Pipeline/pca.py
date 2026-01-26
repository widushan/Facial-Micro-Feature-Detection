import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import mediapipe as mp
import glob

# Ensure we can import pd.py from the current directory
sys.path.append(os.getcwd())
try:
    import pd as pd_module
except ImportError:
    # Try importing if we are in the parent directory or ensuring the path is correct
    sys.path.append(os.path.join(os.getcwd(), 'Parkinson-Pipeline'))
    try:
        import pd as pd_module
    except ImportError:
        print("Error: Could not import 'pd.py'. Please ensure it is in the same directory or python path.")
        sys.exit(1)

# Define paths
TRAIN_CSV = "pd_features.csv"
TEST_CSV = "pd_test_features.csv"
# Assuming script is run from Parkinson-Pipeline, so .. leads to Facial-Micro-Feature-Detection
# And the videos are in Facial-Micro-Feature-Detection/PD Videos/Testing
TEST_DIR = "../PD Videos/Testing"

def analyze_training_composition():
    print("="*50)
    print(" 1. Training Set Data Composition Analysis")
    print("="*50)
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found. Please ensure training data features are extracted.")
        return None
    
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded Training Data: {len(df)} rows, {len(df.columns)} columns")
    
    # Analyze Class Distribution
    if 'Label' in df.columns:
        counts = df['Label'].value_counts()
        print("\nClass Distribution (Frames):")
        print(counts)
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Label', palette='viridis')
        plt.title("Training Set Class Distribution (Frames)")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.savefig("training_class_distribution_frames.png")
        plt.close()
        
    # Analyze Video Distribution
    if 'Video' in df.columns:
        # Check if Video column includes augmentation suffix (e.g. _aug)
        # We might want to group by base video name if possible, but augmentation is treated as samples
        unique_videos = df['Video'].unique()
        print(f"\nTotal Unique Video Segments (including augmentations): {len(unique_videos)}")
        
        video_labels = df.drop_duplicates('Video')[['Video', 'Label']]
        vid_counts = video_labels['Label'].value_counts()
        print("\nVideo Distribution by Label (Segments):")
        print(vid_counts)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=vid_counts.index, y=vid_counts.values, palette='viridis')
        plt.title("Training Set Video Segments Distribution")
        plt.ylabel("Count of Videos/Segments")
        plt.savefig("training_video_distribution.png")
        plt.close()

    return df

def perform_pca_analysis(df):
    print("\n" + "="*50)
    print(" 2. PCA Analysis Results")
    print("="*50)
    
    if df is None:
        print("No dataframe provided for PCA.")
        return

    # Drop non-feature columns
    drop_cols = ['Video', 'Label', 'Frame']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if not feature_cols:
        print("No feature columns found.")
        return

    X = df[feature_cols].values
    y = df['Label'].values if 'Label' in df.columns else None
    
    # Helper for plots
    unique_labels = list(set(y)) if y is not None else []
    
    print(f"Running PCA on {X.shape[0]} samples and {X.shape[1]} features...")
    
    # 1. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Explained Variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nPCA Explained Variance Ratio (Top 5):")
    for i, var in enumerate(explained_variance[:5]):
        print(f" PC{i+1}: {var:.4f}")
        
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_99 = np.argmax(cumulative_variance >= 0.99) + 1
    print(f"\nNumber of components to explain 95% variance: {n_95}")
    print(f"Number of components to explain 99% variance: {n_99}")
    
    # 4. Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), explained_variance[:20], marker='o', linestyle='--')
    plt.title("PCA Scree Plot (Top 20 Components)")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True)
    plt.savefig("pca_scree_plot.png")
    plt.close()
    print("Saved pca_scree_plot.png")
    
    # 5. 2D Projection Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, style=y, alpha=0.6, palette='bright')
    plt.title(f"PCA 2D Projection\nPC1 ({explained_variance[0]:.2%}) vs PC2 ({explained_variance[1]:.2%})")
    plt.xlabel(f"Principal Component 1")
    plt.ylabel(f"Principal Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("pca_2d_projection.png")
    plt.close()
    print("Saved pca_2d_projection.png")
    
    # 6. Feature Importance (Loadings)
    print("\nTop features contributing to PC1:")
    loadings = pca.components_[0]
    # Sort by absolute value
    indices = np.argsort(np.abs(loadings))[::-1]
    for i in range(10):
        feature_idx = indices[i]
        feature_name = feature_cols[feature_idx]
        loading_val = loadings[feature_idx]
        print(f" {i+1}. {feature_name}: {loading_val:.4f}")

def extract_test_data():
    print("\n" + "="*50)
    print(" Extracting Test Data (No Augmentation) ")
    print("="*50)
    
    if os.path.exists(TEST_CSV):
        print(f"Loaded existing test features from {TEST_CSV}")
        return pd.read_csv(TEST_CSV)
        
    print(f"Processing videos from {TEST_DIR}...")
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory {TEST_DIR} does not exist.")
        return None
        
    data = []
    classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    mp_face_mesh = mp.solutions.face_mesh
    
    for category in classes:
        print(f"Processing Test Category: {category}")
        cat_path = os.path.join(TEST_DIR, category)
        video_files = glob.glob(os.path.join(cat_path, "*.mp4"))
        
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            # print(f"  Processing {video_name}...", end='\r')
            
            # CRITICAL: Reset buffers for each new video as per pd.py logic
            pd_module.reset_buffers()
            
            cap = cv2.VideoCapture(video_path)
            with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, 
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
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
                        
                        # Use pd module's normalization and feature extraction
                        # Note: We access the global _prev_landmarks_global from pd_module
                        lm_list_norm = pd_module.normalize_for_rotation_distance(lm_list, pd_module._prev_landmarks_global)
                        
                        features_dict = {}
                        features_dict.update(pd_module.compute_brow_features(lm_list_norm, pd_module._prev_landmarks_global))
                        features_dict.update(pd_module.compute_cheek_features(lm_list_norm, pd_module._prev_landmarks_global))
                        features_dict.update(pd_module.compute_eye_features(lm_list_norm, pd_module._prev_landmarks_global))
                        features_dict.update(pd_module.compute_jaw_features(lm_list_norm, pd_module._prev_landmarks_global))
                        features_dict.update(pd_module.compute_lips_features(lm_list_norm, pd_module._prev_landmarks_global))
                        features_dict.update(pd_module.compute_mouth_features(lm_list_norm, pd_module._prev_landmarks_global))
                        
                        features_dict['Video'] = video_name
                        features_dict['Label'] = category
                        features_dict['Frame'] = frame_count
                        data.append(features_dict)
                        
                        # Update global prev landmarks in pd_module
                        pd_module._prev_landmarks_global = lm_list_norm
            cap.release()
    
    print("\nFeature extraction complete.")
    if not data:
        print("No data extracted. Please check video paths.")
        return None
        
    df = pd.DataFrame(data)
    df = df.fillna(0)
    df.to_csv(TEST_CSV, index=False)
    print(f"Test features saved to {TEST_CSV}")
    return df

def analyze_test_composition(df):
    print("\n" + "="*50)
    print(" 3. Test Set Data Composition Analysis")
    print("="*50)
    
    if df is None:
        print("No test data provided.")
        return

    print(f"Total Test Samples: {len(df)}")
    
    if 'Label' in df.columns:
        counts = df['Label'].value_counts()
        print("\nTest Class Distribution (Frames):")
        print(counts)
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Label', palette='coolwarm')
        plt.title("Test Set Class Distribution (Frames)")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.savefig("test_class_distribution_frames.png")
        plt.close()
        
    if 'Video' in df.columns:
        unique_videos = df['Video'].unique()
        print(f"\nTotal Unique Test Videos: {len(unique_videos)}")
        
        video_labels = df.drop_duplicates('Video')[['Video', 'Label']]
        vid_counts = video_labels['Label'].value_counts()
        print("\nTest Video Distribution by Label:")
        print(vid_counts)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=vid_counts.index, y=vid_counts.values, palette='coolwarm')
        plt.title("Test Set Video Distribution")
        plt.ylabel("Count of Videos")
        plt.savefig("test_video_distribution.png")
        plt.close()

if __name__ == "__main__":
    # 1. Training Analysis & 2. PCA
    train_df = analyze_training_composition()
    if train_df is not None:
        perform_pca_analysis(train_df)
    
    # 3. Test Composition
    test_df = extract_test_data()
    if test_df is not None:
        analyze_test_composition(test_df)
