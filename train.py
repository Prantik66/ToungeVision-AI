import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

from dataset import TongueDataset
from rbfnn import RBFNN


# -----------------------------
# device
# -----------------------------
device = torch.device("cpu")
print("Using device:", device)


# -----------------------------
# load dataset
# -----------------------------
dataset = TongueDataset("dataset")
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

print("Total images:", len(dataset))


# -----------------------------
# load pretrained ResNet50
# -----------------------------
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# remove classification layer
feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()
feature_extractor.to(device)

print("ResNet50 loaded.")


# -----------------------------
# extract features
# -----------------------------
all_features = []
all_labels = []

print("Extracting features...")

with torch.no_grad():
    for images, labels in tqdm(loader):

        images = images.to(device)

        features = feature_extractor(images)

        # flatten [batch,2048,1,1] → [batch,2048]
        features = features.view(features.size(0), -1)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())


# combine all batches
X = np.vstack(all_features)
y = np.hstack(all_labels)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

np.save("outputs/features.npy", X)
np.save("outputs/labels.npy", y)

print("Features saved.")

#5-fold cross validation
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

print("\nStarting 5-Fold Cross Validation...")

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = []
fold = 1

all_true = []
all_pred = []

for train_idx, test_idx in kf.split(X, y):

    print(f"\nFold {fold}")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # normalize
    max_val = np.max(np.abs(X_train))
    X_train = X_train / max_val
    X_test = X_test / max_val

    # train
    rbf = RBFNN(num_centers=150, sigma=5.0)
    rbf.fit(X_train, y_train)

    # predict
    y_pred = rbf.predict(X_test)

    # accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    cv_scores.append(acc)

    all_true.extend(y_test)
    all_pred.extend(y_pred)

    fold += 1


print("\n===== FINAL CV RESULTS =====")
print("Fold Accuracies:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
print("Std Dev:", np.std(cv_scores))
