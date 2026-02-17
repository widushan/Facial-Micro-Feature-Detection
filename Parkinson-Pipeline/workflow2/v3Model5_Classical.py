import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import joblib

# Paths
HEALTHY_CSV = "left_eye_magnitude_healthy.csv"
PARKINSON_CSV = "left_eye_magnitude_parkinson.csv"
OUTPUT_DIR = "model_validation_classical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features(signal, fps=20):
    """
    Extract statistical and frequency-domain features from a raw signal.
    """
    signal = np.array(signal)
    
    # Statistical Features
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    range_val = max_val - min_val
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)
    rms = np.sqrt(np.mean(signal**2))
    
    # Temporal Features
    zero_crossings = np.where(np.diff(np.sign(signal - mean_val)))[0]
    zcr = len(zero_crossings) / len(signal)
    abs_change = np.mean(np.abs(np.diff(signal)))
    
    # Frequency Features (FFT)
    n = len(signal)
    yf = fft(signal - mean_val)
    xf = fftfreq(n, 1 / fps)
    
    # Take positive frequencies
    pos_mask = xf >= 0
    xf = xf[pos_mask]
    yf = np.abs(yf[pos_mask])
    
    # Top 3 peak frequencies and their magnitudes
    peak_indices = np.argsort(yf)[-3:][::-1]
    peak_freqs = xf[peak_indices]
    peak_mags = yf[peak_indices]
    
    features = {
        'mean': mean_val,
        'std': std_val,
        'max': max_val,
        'min': min_val,
        'range': range_val,
        'skew': skewness,
        'kurt': kurtosis,
        'rms': rms,
        'zcr': zcr,
        'abs_change': abs_change,
        'peak_f1': peak_freqs[0] if len(peak_freqs) > 0 else 0,
        'peak_m1': peak_mags[0] if len(peak_mags) > 0 else 0,
        'peak_f2': peak_freqs[1] if len(peak_freqs) > 1 else 0,
        'peak_m2': peak_mags[1] if len(peak_mags) > 1 else 0,
    }
    
    return features

print("Loading data...")
df_healthy = pd.read_csv(HEALTHY_CSV)
df_parkinson = pd.read_csv(PARKINSON_CSV)

def build_feature_df(df, label):
    data = []
    videos = df['Video'].unique()
    for vid in videos:
        vid_df = df[df['Video'] == vid].sort_values('Frame')
        signal = vid_df['Left_Eye_Mag'].values
        feats = extract_features(signal)
        feats['Label'] = label
        feats['Video'] = vid
        data.append(feats)
    return pd.DataFrame(data)

full_feat_df = pd.concat([
    build_feature_df(df_healthy, 0),
    build_feature_df(df_parkinson, 1)
]).reset_index(drop=True)

X = full_feat_df.drop(['Label', 'Video'], axis=1)
y = full_feat_df['Label']

print(f"Feature set ready: {X.shape[0]} samples, {X.shape[1]} features.")
print(f"Class distribution: {y.value_counts().to_dict()}")

# 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
all_true = []
all_probs = []
all_preds = []

print("\nStarting Cross-Validation (Random Forest)...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Initialize and train RF
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Predictions
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    fold_accuracies.append(acc)
    
    all_true.extend(y_test)
    all_probs.extend(probs)
    all_preds.extend(preds)
    
    print(f"Fold {fold+1}/5 | Accuracy: {acc:.4f}")

# Aggregate Results
print(f"\nAverage CV Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

# Final Reporting
report = classification_report(all_true, all_preds, target_names=['Healthy', 'Parkinson'], digits=4)
print("\nAggregated Classical ML Report:\n")
print(report)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Parkinson'])
disp.plot(cmap=plt.cm.Greens)
plt.title('Classical ML Confusion Matrix (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(all_true, all_probs)
roc_auc = roc_auc_score(all_true, all_probs)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp.plot()
plt.title(f'Classical ML ROC Curve (AUC = {roc_auc:.4f})')
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
plt.close()

# Save the final model (trained on all data)
final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
final_model.fit(X, y)
joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'random_forest_parkinson_v1.joblib'))

print(f"\nAll results saved to: {OUTPUT_DIR}")
