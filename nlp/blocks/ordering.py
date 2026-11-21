"""Directed next-turn scores -> permutation. Beam search is approximate, O(B*n^3)."""
import numpy as np
import torch
from torch import nn


class PairScorer(nn.Module):
    """Directional pair scorer on cached utterance or sentence embeddings."""
    def __init__(self, dimensions, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dimensions, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, first, second):
        return self.net(torch.cat([first, second, first * second, second - first], dim=-1)).squeeze(-1)


def beam_order(scores, prefix=(), width=32):
    """scores[i,j] rewards j immediately following i. Supply known first turns."""
    scores = np.asarray(scores)
    n = len(scores)
    prefix = tuple(prefix)
    if scores.shape != (n, n) or len(set(prefix)) != len(prefix) or any(i < 0 or i >= n for i in prefix):
        raise ValueError("Need square scores and distinct in-range prefix IDs")
    if width < 1 or n == 0:
        raise ValueError("Need nonempty scores and positive beam width")
    beams = [(sum(scores[i, j] for i, j in zip(prefix, prefix[1:])), prefix)]
    for _ in range(n - len(prefix)):
        expanded = []
        for score, order in beams:
            for nxt in set(range(n)) - set(order):
                edge = scores[order[-1], nxt] if order else 0.0
                expanded.append((score + edge, order + (nxt,)))
        beams = sorted(expanded, key=lambda x: (-x[0], x[1]))[:width]
    return list(beams[0][1])


def pairwise_order_accuracy(predicted_ranks, true_ranks):
    predicted_ranks, true_ranks = map(np.asarray, (predicted_ranks, true_ranks))
    n = len(true_ranks)
    if sorted(predicted_ranks.tolist()) != list(range(n)):
        return 0.0
    i, j = np.triu_indices(n, 1)
    return float(np.mean((predicted_ranks[i] < predicted_ranks[j]) == (true_ranks[i] < true_ranks[j]))) if n > 1 else 1.0
