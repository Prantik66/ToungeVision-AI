import numpy as np
from sklearn.cluster import KMeans


class RBFNN:
    def __init__(self, num_centers=100, sigma=5.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    # -----------------------------
    # Gaussian RBF
    # -----------------------------
    def _rbf(self, X, centers):
        """
        X: (N, D)
        centers: (K, D)
        return: (N, K)
        """
        N = X.shape[0]
        K = centers.shape[0]

        G = np.zeros((N, K))

        for i in range(K):
            diff = X - centers[i]
            G[:, i] = np.exp(-np.sum(diff**2, axis=1) / (2 * self.sigma**2))

        return G

    # -----------------------------
    # Training
    # -----------------------------
    def fit(self, X, y):
        print("Running KMeans for centers...")

        # Step 1: find centers
        kmeans = KMeans(n_clusters=self.num_centers, random_state=0)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        print("Centers shape:", self.centers.shape)

        # Step 2: compute RBF matrix
        G = self._rbf(X, self.centers)

        print("RBF matrix shape:", G.shape)

        # Step 3: solve weights (least squares)
        self.weights = np.linalg.pinv(G) @ y

        print("Training complete.")

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, X):
        G = self._rbf(X, self.centers)
        y_pred = G @ self.weights

        # convert to 0/1
        return np.round(y_pred)

    def predict_proba(self, X):
        G = self._rbf(X, self.centers)
        return G @ self.weights