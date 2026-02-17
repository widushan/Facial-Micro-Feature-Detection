import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from sklearn.utils.class_weight import compute_class_weight

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_FOLDS = 5
PATIENCE = 15          # Increased for longer training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECTROGRAM_DIR = "Spectrograms"
OUTPUT_DIR = "model_validation_optimized"
os.makedirs(OUTPUT_DIR, exist_ok=True)


MODEL_TYPE = 'efficientnet' # Switched to efficientnet for better performance

# Custom Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels  # 0: Healthy, 1: Parkinson
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load Data
healthy_dir = os.path.join(SPECTROGRAM_DIR, "Healthy")
parkinson_dir = os.path.join(SPECTROGRAM_DIR, "Parkinson")

if not os.path.exists(healthy_dir) or not os.path.exists(parkinson_dir):
    print(f"Error: Spectrogram directories not found in {SPECTROGRAM_DIR}")
    exit(1)

healthy_paths = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith('.png')]
parkinson_paths = [os.path.join(parkinson_dir, f) for f in os.listdir(parkinson_dir) if f.endswith('.png')]

all_paths = np.array(healthy_paths + parkinson_paths)
all_labels = np.array([0] * len(healthy_paths) + [1] * len(parkinson_paths))

# Robust data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
fold_accuracies = []
fold_aucs = []
best_model_state = None
best_acc = 0.0

print(f"Using {DEVICE} for training.")
print(f"Model selected: {MODEL_TYPE}")

for fold, (train_idx, test_idx) in enumerate(skf.split(all_paths, all_labels)):
    print(f"\nFold {fold+1}/{NUM_FOLDS}")
    train_paths_fold, test_paths_fold = all_paths[train_idx], all_paths[test_idx]
    train_labels_fold, test_labels_fold = all_labels[train_idx], all_labels[test_idx]

    # Datasets & Loaders
    train_dataset = SpectrogramDataset(train_paths_fold, train_labels_fold, train_transform)
    test_dataset = SpectrogramDataset(test_paths_fold, test_labels_fold, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute Class Weights for the current fold
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_fold), y=train_labels_fold)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # Model Initialization
    if MODEL_TYPE == 'efficientnet':
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 2)
        )
    elif MODEL_TYPE == 'mobilenet':
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 2)
        )
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 2)
        )
        
    model = model.to(DEVICE)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Early stopping variables
    early_stop_counter = 0
    min_val_loss = float('inf')

    train_losses, test_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        test_loss = 0.0
        preds_prob, preds_label, true = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                labels_pred = np.argmax(outputs.cpu().numpy(), axis=1)
                preds_prob.extend(probs)
                preds_label.extend(labels_pred)
                true.extend(labels.cpu().numpy())
        test_losses.append(test_loss / len(test_loader))

        scheduler.step()

        acc = accuracy_score(true, preds_label)
        auc = roc_auc_score(true, preds_prob)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {test_losses[-1]:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

        # Early stopping & best model tracking
        if test_losses[-1] < min_val_loss:
            min_val_loss = test_losses[-1]
            early_stop_counter = 0
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    fold_accuracies.append(acc)
    fold_aucs.append(auc)

# Summary of cross-validation
print(f"\nCross-Validation Results ({NUM_FOLDS}-fold):")
print(f"Average Test Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Average Test AUC:      {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# Save best model found across folds
if best_model_state:
    model_save_path = os.path.join(OUTPUT_DIR, f'best_{MODEL_TYPE}_optimized.pth')
    torch.save(best_model_state, model_save_path)
    print(f"Best model saved to {model_save_path}")

# --------------------------------------------------------------------
# Final evaluation using the best model on the last fold's test set
# --------------------------------------------------------------------
model.load_state_dict(best_model_state)
model.eval()

with torch.no_grad():
    preds_prob, preds_label, true = [], [], []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        labels_pred = np.argmax(outputs.cpu().numpy(), axis=1)
        preds_prob.extend(probs)
        preds_label.extend(labels_pred)
        true.extend(labels.cpu().numpy())

# Reports & Charts
# 1. Classification Report
report = classification_report(true, preds_label, target_names=['Healthy', 'Parkinson'], digits=4)
print("\nFinal Classification Report (last fold):\n")
print(report)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# 2. Confusion Matrix
cm = confusion_matrix(true, preds_label)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Parkinson'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix ({MODEL_TYPE} - optimized)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(true, preds_prob)
roc_auc = roc_auc_score(true, preds_prob)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp.plot()
plt.title(f'ROC Curve ({MODEL_TYPE} - optimized)')
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
plt.close()

# 4. Loss Curves (last fold)
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curves ({MODEL_TYPE} – Optimized)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'))
plt.close()

# 5. Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
precision, recall, _ = precision_recall_curve(true, preds_prob)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title(f'Precision-Recall Curve ({MODEL_TYPE} - optimized)')
plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'))
plt.close()

print(f"\nAll validation reports and plots saved to: {OUTPUT_DIR}")
