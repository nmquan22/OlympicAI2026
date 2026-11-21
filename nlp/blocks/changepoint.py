"""One human->machine boundary; original character offsets are never normalized."""
import re
import numpy as np


def sentence_spans(text):
    """Heuristic sentence slices. Abbreviations may require a task-specific segmenter."""
    starts = [0] + [m.end() for m in re.finditer(r"[.!?][\"'\u201d]*\s+", text)]
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(text)]) if text[a:b].strip()]


def locate_boundary(machine_probabilities, starts, min_left=1, min_right=1):
    """Maximum-likelihood single change. Score a boundary BEFORE sentence k."""
    p = np.asarray(machine_probabilities).clip(1e-6, 1 - 1e-6)
    candidates = np.arange(min_left, len(p) - min_right + 1)
    if len(candidates) == 0 or len(starts) != len(p):
        raise ValueError("Need enough sentences on both sides and aligned offsets")
    left = np.r_[0, np.cumsum(np.log1p(-p))]
    right = np.r_[np.cumsum(np.log(p)[::-1])[::-1], 0]
    best = candidates[np.argmax(left[candidates] + right[candidates])]
    return int(starts[best])


def boundary_score(predicted, truth, tau=100):
    return float(np.exp(-np.abs(np.asarray(predicted) - np.asarray(truth)) / tau).mean())
