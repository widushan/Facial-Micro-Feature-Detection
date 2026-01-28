"""
PCA Analysis and Dataset Composition Tool (No Augmentation)
Analyzes the extracted features from no_augment_pd.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter

# Configuration - Match no_augment_pd.py settings
OUTPUT_DIR = "."
ALL_FEATURES_CSV = "pd_features_no_aug.csv"
SEQ_LENGTH = 240
NUM_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data():
    """Load the extracted features from CSV"""
    csv_path = os.path.join(OUTPUT_DIR, ALL_FEATURES_CSV)
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run no_augment_pd.py first to extract features.")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df

def get_feature_columns(df):
    """Get feature column names (exclude metadata columns)"""
    drop_cols = ['Video', 'Label', 'Frame']
    return [c for c in df.columns if c not in drop_cols]

def analyze_dataset_composition(df):
    """Analyze overall dataset composition"""
    print("\n" + "="*60)
    print("DATASET COMPOSITION ANALYSIS")
    print("="*60)
    
    # Basic stats
    print(f"\nTotal samples (frames): {len(df):,}")
    print(f"Total features: {len(get_feature_columns(df))}")
    
    # Label distribution
    print("\n--- Label Distribution ---")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} samples ({pct:.1f}%)")
    
    # Video distribution
    print("\n--- Video Distribution ---")
    video_counts = df.groupby('Label')['Video'].nunique()
    for label, count in video_counts.items():
        print(f"  {label}: {count} unique videos")
    
    # Check for augmented videos
    all_videos = df['Video'].unique()
    original_videos = [v for v in all_videos if '_aug' not in v]
    augmented_videos = [v for v in all_videos if '_aug' in v]
    
    print(f"\n--- Augmentation Stats ---")
    print(f"  Original videos: {len(original_videos)}")
    print(f"  Augmented versions: {len(augmented_videos)}")
    print(f"  Augmentation ratio: {len(augmented_videos) / max(len(original_videos), 1):.1f}x")
    
    # Frames per video stats
    frames_per_video = df.groupby('Video').size()
    print(f"\n--- Frames per Video ---")
    print(f"  Mean: {frames_per_video.mean():.1f}")
    print(f"  Std: {frames_per_video.std():.1f}")
    print(f"  Min: {frames_per_video.min()}")
    print(f"  Max: {frames_per_video.max()}")
    
    return label_counts

def analyze_train_test_split(df, test_size=TEST_SIZE):
    """Analyze train/test split composition"""
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT ANALYSIS")
    print("="*60)
    
    feature_cols = get_feature_columns(df)
    
    # Group by video (to avoid data leakage)
    grouped = df.groupby('Video')
    video_names = list(grouped.groups.keys())
    video_labels = [grouped.get_group(v)['Label'].iloc[0] for v in video_names]
    
    # Stratified split
    train_videos, test_videos, train_labels, test_labels = train_test_split(
        video_names, video_labels, 
        test_size=test_size, 
        stratify=video_labels,
        random_state=RANDOM_STATE
    )
    
    # Create train/test dataframes
    df_train = df[df['Video'].isin(train_videos)]
    df_test = df[df['Video'].isin(test_videos)]
    
    print(f"\n--- Training Set ---")
    print(f"  Videos: {len(train_videos)}")
    print(f"  Samples (frames): {len(df_train):,}")
    train_label_counts = Counter(train_labels)
    for label, count in sorted(train_label_counts.items()):
        pct = count / len(train_videos) * 100
        print(f"    {label}: {count} videos ({pct:.1f}%)")
    
    print(f"\n--- Test Set ---")
    print(f"  Videos: {len(test_videos)}")
    print(f"  Samples (frames): {len(df_test):,}")
    test_label_counts = Counter(test_labels)
    for label, count in sorted(test_label_counts.items()):
        pct = count / len(test_videos) * 100
        print(f"    {label}: {count} videos ({pct:.1f}%)")
    
    return df_train, df_test, train_videos, test_videos

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
        # Plotting moved to dashboard
        
    # Analyze Video Distribution
    if 'Video' in df.columns:
        unique_videos = df['Video'].unique()
        print(f"\nTotal Unique Video Segments (including augmentations): {len(unique_videos)}")
        
        video_labels = df.drop_duplicates('Video')[['Video', 'Label']]
        vid_counts = video_labels['Label'].value_counts()
        print("\nVideo Distribution by Label (Segments):")
        print(vid_counts)
        # Plotting moved to dashboard or separate call if needed

    return df

def analyze_cross_validation_folds(df, n_folds=NUM_FOLDS):
    """Analyze cross-validation fold composition"""
    print("\n" + "="*60)
    print(f"{n_folds}-FOLD CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    # Group by video
    grouped = df.groupby('Video')
    video_names = np.array(list(grouped.groups.keys()))
    video_labels = np.array([grouped.get_group(v)['Label'].iloc[0] for v in video_names])
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(video_names, video_labels)):
        train_vids = video_names[train_idx]
        val_vids = video_names[val_idx]
        train_labs = video_labels[train_idx]
        val_labs = video_labels[val_idx]
        
        print(f"\n--- Fold {fold+1} ---")
        print(f"  Train: {len(train_vids)} videos | Val: {len(val_vids)} videos")
        
        train_counts = Counter(train_labs)
        val_counts = Counter(val_labs)
        
        print(f"  Train distribution: {dict(train_counts)}")
        print(f"  Val distribution: {dict(val_counts)}")

def perform_pca_analysis(df):
    print("\n" + "="*50)
    print(" 2. PCA Analysis Results")
    print("="*50)
    
    if df is None:
        print("No dataframe provided for PCA.")
        return None, None, [], None

    # Drop non-feature columns
    drop_cols = ['Video', 'Label', 'Frame']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if not feature_cols:
        print("No feature columns found.")
        return None, None, [], None

    X = df[feature_cols].values
    
    # 1. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Explained Variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nNumber of components to explain 95% variance: {n_95}")
    
    # Print Top 5 Variance
    print("\nPCA Explained Variance Ratio (Top 5):")
    for i, var in enumerate(explained_variance[:5]):
        print(f" PC{i+1}: {var:.4f}")
        
    # 4. Feature Importance (Loadings) - Print to console
    print("\nTop features contributing to PC1:")
    loadings = pca.components_[0]
    indices = np.argsort(np.abs(loadings))[::-1]
    for i in range(10):
        if i < len(indices):
            feature_idx = indices[i]
            feature_name = feature_cols[feature_idx]
            loading_val = loadings[feature_idx]
            print(f" {i+1}. {feature_name}: {loading_val:.4f}")
            
    return pca, X_scaled, feature_cols, cumulative_variance

def get_top_pca_features(components, feature_names, top_n=15):
    """Get features with highest loadings across first N components"""
    # Aggregate absolute loadings
    total_contribution = np.sum(np.abs(components), axis=0)
    
    # Sort by contribution
    indices = np.argsort(total_contribution)[::-1][:top_n]
    
    return [(feature_names[i], total_contribution[i]) for i in indices]

def select_features_via_pca(X, feature_names, variance_threshold=0.99):
    """Select features using PCA (replicates pd.py logic)"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=variance_threshold)
    pca.fit(X_scaled)
    
    selected_indices = set()
    components = pca.components_
    
    for i in range(components.shape[0]):
        # Get top 10 features contributing to this component
        top_indices = np.argsort(np.abs(components[i]))[-10:]
        for idx in top_indices:
            selected_indices.add(idx)
    
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features

def plot_pca_results(pca, cumulative_var, df, X_scaled, save_path="pca_analysis_dashboard_no_aug.png"):
    """Generate PCA visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scree plot
    ax1 = axes[0, 0]
    n_show = min(30, len(pca.explained_variance_ratio_))
    ax1.bar(range(1, n_show+1), pca.explained_variance_ratio_[:n_show], alpha=0.7, label='Individual')
    ax1.plot(range(1, n_show+1), cumulative_var[:n_show], 'r-o', markersize=4, label='Cumulative')
    ax1.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% threshold')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Scree Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative variance (zoomed)
    ax2 = axes[0, 1]
    ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 'b-')
    ax2.axhline(y=0.80, color='g', linestyle='--', alpha=0.7, label='80%')
    ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 2D PCA projection
    ax3 = axes[1, 0]
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)
    
    labels = df['Label'].values
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax3.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=label, alpha=0.5, s=10)
    
    ax3.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_title('2D PCA Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Label distribution
    ax4 = axes[1, 1]
    label_counts = df['Label'].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(label_counts)))
    bars = ax4.bar(label_counts.index, label_counts.values, color=colors)
    ax4.set_xlabel('Label')
    ax4.set_ylabel('Sample Count')
    ax4.set_title('Dataset Label Distribution')
    
    # Add count labels on bars
    for bar, count in zip(bars, label_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()

def main():
    print("="*60)
    print("PD FACE DATASET ANALYSIS TOOL")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # 1. Dataset Composition Analysis
    analyze_dataset_composition(df)
    
    # 2. Train/Test Split Analysis
    df_train, df_test, train_videos, test_videos = analyze_train_test_split(df)
    
    # 3. Cross-Validation Fold Analysis
    analyze_cross_validation_folds(df)
    
    # 4. PCA Analysis
    pca, X_scaled, feature_cols, cumulative_var = perform_pca_analysis(df)
    
    # 5. Feature Selection (replicate pd.py logic)
    print("\n" + "="*60)
    print("FEATURE SELECTION (PCA-based)")
    print("="*60)
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    selected_features = select_features_via_pca(X, feature_cols)
    
    print(f"\nSelected {len(selected_features)} features out of {len(feature_cols)} total")
    print(f"Reduction: {(1 - len(selected_features)/len(feature_cols))*100:.1f}%")
    print(f"\nSelected features:")
    for i, feat in enumerate(sorted(selected_features), 1):
        print(f"  {i:2d}. {feat}")
    
    # 6. Generate plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    plot_pca_results(pca, cumulative_var, df, X_scaled)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
