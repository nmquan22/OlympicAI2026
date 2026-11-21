"""Small training utilities. No downloads, hidden device selection or side effects."""
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch


def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Budget:
    """Call can_start before an epoch; reserve time for writing predictions."""
    seconds: float
    reserve: float = 30.0
    started: float = field(default_factory=time.monotonic)

    @property
    def remaining(self):
        return self.seconds - (time.monotonic() - self.started)

    def can_start(self, estimated_seconds):
        return self.remaining > estimated_seconds + self.reserve


def distillation_loss(student, teacher, labels, temperature=2.0, soft_weight=0.5):
    """Teacher logits must use exactly the same class order as the student."""
    from torch.nn import functional as F
    hard = F.cross_entropy(student, labels)
    soft = F.kl_div((student / temperature).log_softmax(-1),
                    (teacher.detach() / temperature).softmax(-1), reduction="batchmean")
    return (1 - soft_weight) * hard + soft_weight * temperature**2 * soft
