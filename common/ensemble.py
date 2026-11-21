"""Tune using validation predictions with identical sample and class orders."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss


def fit_temperature(logits, labels):
    """Fit on a calibration split, then freeze for final evaluation."""
    classes = np.arange(logits.shape[1])
    objective = lambda log_t: log_loss(labels, softmax(logits / np.exp(log_t), axis=1), labels=classes)
    return float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))


def blend_probabilities(predictions, weights):
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be nonnegative with positive total")
    return np.average(np.stack(predictions), axis=0, weights=weights)


def disagreement(predictions):
    """Fraction of model pairs disagreeing, for each sample. Shape [models, samples]."""
    predictions = np.asarray(predictions)
    models = len(predictions)
    if models < 2:
        raise ValueError("Supply at least two models")
    return sum(predictions[i] != predictions[j] for i in range(models) for j in range(i)) / (models * (models - 1) / 2)
