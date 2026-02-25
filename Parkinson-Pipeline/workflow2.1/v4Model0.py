import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

# ===============================
# 1. Reproducibility
# ===============================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 2. Parameters
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 16 
EPOCHS = 30       # Increased for consolidated dataset
LEARNING_RATE = 2e-4
NUM_FOLDS = 5
PATIENCE = 7

FEATURE_ROOT = "Feature_Analysis"
OUTPUT_DIR = "Consolidated_Model_Results_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LIST = ["mobilenet_v3", "efficientnet_b0", "resnet18"]

print(f"Using device: {DEVICE}")

# ===============================
# 3. Dataset
# ===============================
class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ===============================
# 4. Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ===============================
# 5. Model Builder
# ===============================
def build_model(model_name):
    if model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        num_ftrs = model.classifier[0].in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        model.classifier = head

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        model.classifier = head

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        model.fc = head

    return model.to(DEVICE)

# ===============================
# 6. Data Aggregation & Video Mapping
# ===============================
print("\n[STEP 1] Aggregating all features...")
data_records = []
feature_dirs = [d for d in os.listdir(FEATURE_ROOT) if os.path.isdir(os.path.join(FEATURE_ROOT, d))]

for feat in feature_dirs:
    spec_path = os.path.join(FEATURE_ROOT, feat, "Spectrograms")
    if not os.path.exists(spec_path): continue

    for category in ["Healthy", "Parkinson"]:
        cat_dir = os.path.join(spec_path, category)
        if not os.path.exists(cat_dir): continue
        
        label = 0 if category == "Healthy" else 1
        for f in os.listdir(cat_dir):
            if f.endswith(".png"):
                # Video name is before '_mel.png' or similar suffix
                video_name = f.split("_mel")[0]
                data_records.append({
                    'path': os.path.join(cat_dir, f),
                    'label': label,
                    'video': video_name,
                    'feature': feat
                })

df = pd.DataFrame(data_records)
if df.empty:
    print(f"Error: No spectrograms found in {FEATURE_ROOT}. Run v4.py first.")
    exit()

all_paths = df['path'].values
all_labels = df['label'].values
all_videos = df['video'].values # Groups for splitting

print(f"Total Combined Images: {len(df)}")
print(f"Unique Source Videos: {df['video'].nunique()}")
print(f"Healthy Images: {len(df[df['label']==0])}, PD Images: {len(df[df['label']==1])}")

# ===============================
# 7. Training Strategy: Global Models
# ===============================
best_overall_acc = 0
results_summary = []

# StratifiedGroupKFold ensures:
# 1. Balanced labels across folds
# 2. Images from the same VIDEO stay together in the same fold (No Leakage)
sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)

for MODEL_TYPE in MODEL_LIST:
    print(f"\n" + "="*50)
    print(f" TRAINING GLOBAL MODEL: {MODEL_TYPE} ")
    print("="*50)

    oof_preds_prob = np.zeros(len(all_labels))
    oof_preds_label = np.zeros(len(all_labels))
    
    # Track the best weights across all folds for this model type
    best_weights_for_model = None
    best_acc_for_model = 0

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(all_paths, all_labels, groups=all_videos)):
        print(f"\n--- Fold {fold+1}/{NUM_FOLDS} ---")
        
        train_paths, val_paths = all_paths[train_idx], all_paths[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

        train_dataset = SpectrogramDataset(train_paths, train_labels, transform)
        val_dataset = SpectrogramDataset(val_paths, val_labels, transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type=="cuda"))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=="cuda"))

        model = build_model(MODEL_TYPE)
        
        cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(DEVICE))
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_v_loss = float('inf')
        patience_cnt = 0
        best_fold_state = None

        for epoch in range(EPOCHS):
            model.train()
            t_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            model.eval()
            v_loss = 0
            v_pred = []
            v_true = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    out = model(inputs)
                    v_loss += criterion(out, labels).item()
                    v_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                    v_true.extend(labels.cpu().numpy())
            
            v_loss /= len(val_loader)
            t_loss /= len(train_loader)
            scheduler.step(v_loss)
            cur_acc = accuracy_score(v_true, v_pred)

            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"  Ep {epoch+1:02d} | T-Loss: {t_loss:.3f} | V-Loss: {v_loss:.3f} | V-Acc: {cur_acc:.3f}")

            if v_loss < best_v_loss:
                best_v_loss = v_loss
                best_fold_state = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE: break

        # Evaluation of Best Fold State
        model.load_state_dict(best_fold_state)
        model.eval()
        f_prob = []
        f_label = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                out = model(inputs.to(DEVICE))
                f_prob.extend(torch.softmax(out, dim=1)[:,1].cpu().numpy())
                f_label.extend(torch.argmax(out, dim=1).cpu().numpy())
        
        oof_preds_prob[val_idx] = f_prob
        oof_preds_label[val_idx] = f_label
        f_acc = accuracy_score(all_labels[val_idx], f_label)
        print(f">> Fold Summary: Acc = {f_acc:.3f}")
        
        if f_acc > best_acc_for_model:
            best_acc_for_model = f_acc
            best_weights_for_model = best_fold_state

    # Global Model Performance (OOF)
    final_acc = accuracy_score(all_labels, oof_preds_label)
    final_auc = roc_auc_score(all_labels, oof_preds_prob)
    print(f"\n[GLOBAL SUMMARY] {MODEL_TYPE} -> Combined OOF Acc: {final_acc:.4f}, AUC: {final_auc:.4f}")

    # Metrics Plotting
    cm = confusion_matrix(all_labels, oof_preds_label)
    ConfusionMatrixDisplay(cm, display_labels=["Healthy", "PD"]).plot(cmap='Blues')
    plt.title(f"Global {MODEL_TYPE} Confusion Matrix\nAcc: {final_acc:.2f}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_confusion_matrix.png"))
    plt.close()

    # Track overall best
    if final_acc > best_overall_acc:
        best_overall_acc = final_acc
        
        # Save absolute best weights
        torch.save(best_weights_for_model, os.path.join(OUTPUT_DIR, "best_global_model.pth"))
        
        # Save best global report
        report = classification_report(all_labels, oof_preds_label, target_names=["Healthy", "Parkinson"])
        with open(os.path.join(OUTPUT_DIR, "best_model_report.txt"), "w") as f:
            f.write(f"Best Architecture: {MODEL_TYPE}\nCombined Accuracy: {final_acc:.4f}\nAUC: {final_auc:.4f}\n\n{report}")

    results_summary.append({
        'Model': MODEL_TYPE,
        'OOF_Accuracy': final_acc,
        'OOF_AUC': final_auc
    })

# Final Summary Table
print("\n" + "="*50)
print(" FINAL RANKING (CONSOLIDATED) ")
print("="*50)
summary_df = pd.DataFrame(results_summary).sort_values(by='OOF_Accuracy', ascending=False)
print(summary_df.to_string(index=False))

print(f"\n>>> OVERALL WINNER: {summary_df.iloc[0]['Model']} (Acc: {best_overall_acc:.4f})")
print(f">>> Weights saved to: {os.path.join(OUTPUT_DIR, 'best_global_model.pth')}")
