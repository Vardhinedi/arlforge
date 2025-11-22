# arlforge/utils/normalizer.py
import numpy as np

class Normalizer:
    """
    Running mean/std normalizer for states (per-dimension).
    Keeps numerical stability for training.
    """
    def __init__(self, size, eps=1e-4, clip=10.0):
        self.n = 0
        self.mean = np.zeros(size, dtype=np.float32)
        self.S = np.zeros(size, dtype=np.float64)  # for variance
        self.eps = eps
        self.clip = clip

    def update(self, x):
        """
        x: np.array of shape (..., size) or list of states
        """
        arr = np.array(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        for row in arr:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.S += delta * delta2

    def var(self):
        return (self.S / (self.n - 1)) if self.n > 1 else np.ones_like(self.mean)

    def std(self):
        return np.sqrt(self.var()) + self.eps

    def normalize(self, x):
        a = np.array(x, dtype=np.float32)
        if a.ndim == 1:
            return np.clip((a - self.mean) / self.std(), -self.clip, self.clip).astype(np.float32)
        else:
            return np.clip((a - self.mean) / self.std(), -self.clip, self.clip).astype(np.float32)

    def denormalize(self, x_norm):
        a = np.array(x_norm, dtype=np.float32)
        if a.ndim == 1:
            return (a * self.std() + self.mean).astype(np.float32)
        else:
            return (a * self.std() + self.mean).astype(np.float32)
