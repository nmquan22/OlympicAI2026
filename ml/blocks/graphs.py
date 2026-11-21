"""Dense graph baseline for small graphs; use sparse operators for large graphs."""
import torch
from torch import nn
from torch.nn import functional as F


def normalized_adjacency(adjacency):
    """Input nonnegative, undirected [N,N] adjacency without self loops."""
    a = adjacency + torch.eye(len(adjacency), device=adjacency.device, dtype=adjacency.dtype)
    inverse = a.sum(-1).clamp_min(1e-12).rsqrt()
    return inverse[:, None] * a * inverse[None, :]


class GCN(nn.Module):
    def __init__(self, inputs, hidden, classes):
        super().__init__()
        self.first, self.second = nn.Linear(inputs, hidden), nn.Linear(hidden, classes)

    def forward(self, features, adjacency):
        a = normalized_adjacency(adjacency)
        return self.second(a @ F.relu(self.first(a @ features)))


def label_propagation(adjacency, labels_onehot, labelled_mask, steps=30, alpha=0.8):
    """Transductive: full graph is available, only TRAIN labels enter seeds."""
    a = normalized_adjacency(adjacency)
    seeds = labels_onehot * labelled_mask[:, None]
    result = seeds.clone()
    for _ in range(steps):
        result = alpha * (a @ result) + (1 - alpha) * seeds
    return result
