import pickle
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# -----------------------------
# load saved RBFNN model
# -----------------------------
with open("models/rbfnn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

centers = model_data["centers"]
weights = model_data["weights"]
sigma = model_data["sigma"]
max_val = model_data["max_val"]


# -----------------------------
# load ResNet50
# -----------------------------
weights_resnet = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights_resnet)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()


# -----------------------------
# preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# RBF transform
# -----------------------------
def rbf_transform(X, centers, sigma):
    N = X.shape[0]
    K = centers.shape[0]

    G = np.zeros((N, K))

    for i in range(K):
        diff = X - centers[i]
        G[:, i] = np.exp(
            -np.sum(diff**2, axis=1) / (2 * sigma**2)
        )

    return G


# -----------------------------
# predict function
# -----------------------------
def predict_tongue(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(image)
        features = features.view(1, -1)
        features = features.numpy()

    # normalize
    features = features / max_val

    # RBF layer
    G = rbf_transform(features, centers, sigma)

    # output
    score = G @ weights
    score = float(score[0])

    label = 1 if score >= 0.5 else 0

    prob = 1 / (1 + np.exp(-score))

    if prob >= 0.585:
        result = "Diabetic"
        confidence = prob * 100
    else:
        result = "Non-Diabetic"
        confidence = (1 - prob) * 100

    return result, confidence

