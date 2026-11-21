"""Contrastive learning requires genuinely different views, not duplicated tensors."""
import torch
from torch.nn import functional as F


def nt_xent(first, second, temperature=0.2):
    n = first.size(0)
    if n < 2:
        raise ValueError("Contrastive learning needs at least two source images")
    z = F.normalize(torch.cat([first, second]), dim=-1)
    scores = z @ z.T / temperature
    scores = scores.masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), -torch.inf)
    positives = (torch.arange(2*n, device=z.device) + n) % (2*n)
    return F.cross_entropy(scores, positives)


@torch.no_grad()
def update_ema(teacher, student, momentum=0.99):
    """Copy buffers too: otherwise BatchNorm running statistics remain stale."""
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.lerp_(source, 1 - momentum)
    for target, source in zip(teacher.buffers(), student.buffers()):
        target.copy_(source)
