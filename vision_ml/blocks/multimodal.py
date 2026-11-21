"""Simple modality fusion with explicit missingness; no assumption of CLIP weights."""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedFusion(nn.Module):
    def __init__(self, dimensions, width, classes):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d, width) for d in dimensions])
        self.gates = nn.Parameter(torch.zeros(len(dimensions)))
        self.head = nn.Linear(width, classes)

    def forward(self, modalities, available):
        """modalities: list [B,Dm], available: bool [B,M], >=1 per sample."""
        if not available.any(1).all():
            raise ValueError("Every sample needs at least one available modality")
        # Mask input before projection so NaNs in missing modalities cannot propagate.
        projected = torch.stack([project(torch.where(available[:, i:i+1], x, 0.0))
                                 for i, (project, x) in enumerate(zip(self.projections, modalities))], dim=1)
        weights = self.gates.expand(len(available), -1).masked_fill(~available, -torch.inf).softmax(1)
        return self.head((projected * weights.unsqueeze(-1)).sum(1))


def ridge_prototypes(support, labels, queries, classes, regularization=1.0):
    """A few-shot baseline on cached features. Class IDs must be 0..classes-1."""
    support, queries = F.normalize(support, dim=-1), F.normalize(queries, dim=-1)
    target = F.one_hot(labels, classes).to(support.dtype)
    # Dual form: solve Ns x Ns, useful when support is much smaller than feature dim.
    gram = support @ support.T + regularization * torch.eye(len(support), device=support.device, dtype=support.dtype)
    return queries @ support.T @ torch.linalg.solve(gram, target)
