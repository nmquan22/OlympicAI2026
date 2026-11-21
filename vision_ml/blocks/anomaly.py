"""Fit standardization and a patch bank only on normal training images."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class PatchBank:
    def __init__(self, limit=8192, seed=7):
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.scaler = StandardScaler()
        self.neighbors = NearestNeighbors(n_neighbors=1)

    def fit(self, normal):
        # [images, rows, columns, channels] -> [patches, channels]
        flat = normal.reshape(-1, normal.shape[-1])
        indices = self.rng.choice(len(flat), min(len(flat), self.limit), replace=False)
        bank = self.scaler.fit_transform(flat[indices])
        self.neighbors.fit(bank)
        return self

    def score(self, features):
        flat = features.reshape(-1, features.shape[-1])
        distances = self.neighbors.kneighbors(self.scaler.transform(flat))[0][:, 0]
        heatmaps = distances.reshape(features.shape[:-1])
        return heatmaps.reshape(len(features), -1).max(1), heatmaps
