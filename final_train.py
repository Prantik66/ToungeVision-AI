import numpy as np
import pickle

from rbfnn import RBFNN


# load saved features
X = np.load("outputs/features.npy")
y = np.load("outputs/labels.npy")

print("Loaded features:", X.shape)

# normalize
max_val = np.max(np.abs(X))
X = X / max_val

print("Training final model...")

# train on ALL data
rbf = RBFNN(
    num_centers=150,
    sigma=5.0
)

rbf.fit(X, y)

# save model
model_data = {
    "centers": rbf.centers,
    "weights": rbf.weights,
    "sigma": rbf.sigma,
    "max_val": max_val
}

with open("models/rbfnn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model saved.")