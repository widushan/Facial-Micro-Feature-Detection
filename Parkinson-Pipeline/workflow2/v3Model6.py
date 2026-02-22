import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
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
BATCH_SIZE = 16   # smaller batch for small dataset
EPOCHS = 50       # reduce epochs
LEARNING_RATE = 3e-4
NUM_FOLDS = 5
PATIENCE = 10

SPECTROGRAM_DIR = "Spectrograms"
OUTPUT_DIR = "OOF_Model_Comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LIST = ["mobilenet_v3", "efficientnet"]

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
# 4. Load Data
# ===============================
healthy_dir = os.path.join(SPECTROGRAM_DIR, "Healthy")
parkinson_dir = os.path.join(SPECTROGRAM_DIR, "Parkinson")

healthy_paths = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith('.png')]
parkinson_paths = [os.path.join(parkinson_dir, f) for f in os.listdir(parkinson_dir) if f.endswith('.png')]

all_paths = np.array(healthy_paths + parkinson_paths)
all_labels = np.array([0]*len(healthy_paths) + [1]*len(parkinson_paths))

print(f"Total samples: {len(all_paths)}")
print(f"Healthy: {len(healthy_paths)}, Parkinson: {len(parkinson_paths)}")

# ===============================
# 5. Medical-Safe Transform (NO augmentation)
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===============================
# 6. Model Builder
# ===============================
def build_model(model_name):

    if model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, 2)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 2)

    return model.to(DEVICE)

# ===============================
# 7. Cross Validation + OOF
# ===============================
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)

best_global_auc = 0
best_model_name = None

for MODEL_TYPE in MODEL_LIST:

    print(f"\n============================")
    print(f"Training Model: {MODEL_TYPE}")
    print(f"============================")

    fold_accuracies = []
    fold_aucs = []

    oof_preds_prob = np.zeros(len(all_labels))
    oof_preds_label = np.zeros(len(all_labels))

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):

        print(f"\nFold {fold+1}/{NUM_FOLDS}")

        train_paths, val_paths = all_paths[train_idx], all_paths[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

        train_dataset = SpectrogramDataset(train_paths, train_labels, transform)
        val_dataset = SpectrogramDataset(val_paths, val_labels, transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(MODEL_TYPE)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        best_val_loss = float('inf')
        patience_counter = 0
        best_fold_state = None

        for epoch in range(EPOCHS):

            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            preds_prob = []
            preds_label = []
            true = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    prob = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                    pred = torch.argmax(outputs, dim=1).cpu().numpy()

                    preds_prob.extend(prob)
                    preds_label.extend(pred)
                    true.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            acc = accuracy_score(true, preds_label)
            auc = roc_auc_score(true, preds_prob)

            print(f"Epoch {epoch+1:2d} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_fold_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        # Load best fold model
        model.load_state_dict(best_fold_state)

        # OOF predictions
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                prob = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                pred = torch.argmax(outputs, dim=1).cpu().numpy()

                oof_preds_prob[val_idx[:len(prob)]] = prob
                oof_preds_label[val_idx[:len(pred)]] = pred

        fold_accuracies.append(acc)
        fold_aucs.append(auc)

    # ===============================
    # Aggregate OOF Evaluation
    # ===============================
    final_acc = accuracy_score(all_labels, oof_preds_label)
    final_auc = roc_auc_score(all_labels, oof_preds_prob)

    print(f"\nOOF Accuracy: {final_acc:.4f}")
    print(f"OOF AUC: {final_auc:.4f}")

    print("\nClassification Report (Full Dataset OOF):")
    print(classification_report(all_labels, oof_preds_label,
                                target_names=["Healthy","Parkinson"]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, oof_preds_label)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy","Parkinson"])
    disp.plot()
    plt.title(f"{MODEL_TYPE} - OOF Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, oof_preds_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=final_auc).plot()
    plt.title(f"{MODEL_TYPE} - OOF ROC Curve")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_roc.png"))
    plt.close()

    if final_auc > best_global_auc:
        best_global_auc = final_auc
        best_model_name = MODEL_TYPE

print(f"\nBest Model Based on OOF AUC: {best_model_name}")
print(f"Best OOF AUC: {best_global_auc:.4f}")