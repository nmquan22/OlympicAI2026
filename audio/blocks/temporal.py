"""Moment retrieval on cached clip embeddings; timestamps are in seconds."""
import numpy as np


def temporal_iou(first, second):
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / max(union, 1e-12)


def best_interval(scores, boundaries, min_clips=1, max_clips=20, length_penalty=0.0):
    """Maximize sum(score-length_penalty); boundaries has len(scores)+1 entries.

    Calibrate scores (e.g. cosine minus dev threshold) before this decoder.
    With all-positive scores, an unconstrained sum always prefers long segments.
    """
    scores, boundaries = np.asarray(scores), np.asarray(boundaries)
    if len(boundaries) != len(scores) + 1 or not np.all(np.diff(boundaries) > 0):
        raise ValueError("Need strictly increasing clip boundaries")
    prefix = np.r_[0, np.cumsum(scores - length_penalty)]
    candidates = [(prefix[end] - prefix[start], start, end)
                  for start in range(len(scores))
                  for end in range(start + min_clips, min(start + max_clips, len(scores)) + 1)]
    if not candidates:
        raise ValueError("No interval satisfies length limits")
    _, start, end = max(candidates)
    return float(boundaries[start]), float(boundaries[end])
