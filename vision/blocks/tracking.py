"""Associate boxes in two consecutive frames. XYXY uses image edges."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_iou(previous, current):
    a = np.asarray(previous, dtype=float).reshape(-1, 4)
    b = np.asarray(current, dtype=float).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[:, :2])
    high = np.minimum(a[:, None, 2:], b[:, 2:])
    intersection = np.maximum(high - low, 0).prod(-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return intersection / np.maximum(area_a[:, None] + area_b - intersection, 1e-12)


def associate(previous, current, threshold=0.3):
    overlap = pairwise_iou(previous, current)
    # Any invalid edge costs more than the total IoU cost of a valid matching.
    cost = np.where(overlap >= threshold, 1 - overlap, 1 + min(overlap.shape))
    left, right = linear_sum_assignment(cost)
    keep = overlap[left, right] >= threshold
    return list(zip(left[keep].tolist(), right[keep].tolist()))
