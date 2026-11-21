"""Soft semantic comparisons: robust to public/private embedding mismatch."""
import numpy as np
from scipy.special import expit


class ComparisonSearch:
    def __init__(self, embeddings, temperature=0.15):
        x = np.asarray(embeddings, dtype=float)
        self.x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        self.log_weights = np.zeros(len(x))
        self.used = set()
        self.temperature = temperature

    def observe(self, first, second, verdict):
        difference = self.x @ (self.x[first] - self.x[second])
        probabilities = {"first": expit(difference / self.temperature),
                         "second": expit(-difference / self.temperature),
                         "same": np.exp(-np.abs(difference) / self.temperature)}
        self.log_weights += np.log(probabilities[verdict].clip(0.02, 1))
        # Reaching observe means neither proposed word ended the game.
        self.used.update((first, second))

    def propose(self, incumbent, explore_weight=0.35):
        remaining = np.array([i for i in range(len(self.x)) if i not in self.used])
        if len(remaining) == 0:
            raise ValueError("Vocabulary exhausted")
        weights = np.exp(self.log_weights - self.log_weights.max())
        weights[list(self.used)] = 0
        weights /= weights.sum()
        difference = self.x @ (self.x[remaining] - self.x[incumbent]).T
        probability = weights @ expit(difference / self.temperature)
        entropy = -(probability * np.log(probability.clip(1e-12)) +
                    (1 - probability) * np.log((1 - probability).clip(1e-12)))
        return int(remaining[np.argmax(weights[remaining] + explore_weight * entropy)])
