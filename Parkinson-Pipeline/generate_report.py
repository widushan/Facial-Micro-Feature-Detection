
import os
import sys
import numpy as np
import pandas as pd_lib # Renamed to avoid conflict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import local module
# Ensure current dir is in path
sys.path.append(os.getcwd())
try:
    import pd as pd_module
except ImportError:
    print("Error: Could not import 'pd.py'. Run this from the Parkinson-Pipeline directory.")
    sys.exit(1)

# Options
SELECTED_FEATURES_CSV = "pd_selected_features.csv"
MODEL_FILE = "pd_detection_model.pth"
SEQ_LENGTH = 240
NUM_FOLDS = 5
TARGET_FOLD_INDEX = 2 # Fold 3 is index 2 (0, 1, 2)

def generate_report():
    print("Loading data...")
    if not os.path.exists(SELECTED_FEATURES_CSV):
        print(f"Error: {SELECTED_FEATURES_CSV} not found. Run pd.py first.")
        return

    df = pd_lib.read_csv(SELECTED_FEATURES_CSV)
    
    # Extract feature names (excluding metadata)
    drop_cols = ['Video', 'Label', 'Frame']
    feature_names = [c for c in df.columns if c not in drop_cols]
    print(f"Features: {len(feature_names)}")

    # Group sequences exactly as in train_model
    grouped = df.groupby('Video')
    X_seq = []
    y_seq = []
    lengths = []
    unique_labels = sorted(df['Label'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    print(f"Classes: {label_map}")
    
    for video, group in grouped:
        feats = group[feature_names].values
        label = label_map[group['Label'].iloc[0]]
        seq_len = len(feats)
        if seq_len > SEQ_LENGTH:
             feats = feats[:SEQ_LENGTH]
             seq_len = SEQ_LENGTH
        X_seq.append(feats)
        y_seq.append(label)
        lengths.append(seq_len)
        
    X_padded = pd_module.pad_sequences(X_seq, maxlen=SEQ_LENGTH, dtype='float32')
    y_array = np.array(y_seq)
    lengths_array = np.array(lengths)

    # Reconstruct StratifiedKFold split to find Fold 3 validation set
    print(f"Reconstructing split for Fold {TARGET_FOLD_INDEX + 1}...")
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    target_val_idx = None
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_padded, y_array)):
        if fold == TARGET_FOLD_INDEX:
            target_val_idx = val_idx
            break
            
    if target_val_idx is None:
        print("Error interpreting folds.")
        return

    # Prepare Validation Data
    X_val = X_padded[target_val_idx]
    y_val = y_array[target_val_idx]
    l_val = lengths_array[target_val_idx]
    
    val_dataset = pd_module.ExpressionDataset(list(X_val), y_val, l_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pd_module.collate_fn)

    # Load Model
    print(f"Loading model from {MODEL_FILE}...")
    feature_dim = len(feature_names)
    num_classes = len(unique_labels)
    # Architecture params must match pd.py exactly
    model = pd_module.OptimizedCNNLSTM(feature_dim, 64, num_classes, num_layers=3)
    
    try:
        model.load_state_dict(torch.load(MODEL_FILE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("Running Inference...")
    with torch.no_grad():
        for val_X, val_y, val_lengths in val_loader:
            outputs = model(val_X, val_lengths)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(val_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Generate Report
    report = classification_report(all_targets, all_preds, target_names=unique_labels)
    print("\n" + "="*40)
    print(f"Classification Report (Fold {TARGET_FOLD_INDEX+1} Validation Set)")
    print("="*40)
    print(report)
    
    # Save Report
    with open('classification_report_generated.txt', 'w') as f:
        f.write(report)
        
    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f'Confusion Matrix (Fold {TARGET_FOLD_INDEX+1})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    output_png = 'confusion_matrix_generated.png'
    plt.savefig(output_png)
    print(f"Confusion matrix saved to {output_png}")
    print(f"Report saved to classification_report_generated.txt")

if __name__ == "__main__":
    generate_report()
