"""Sound classification and CTC decoding; no pretrained speech weights required."""
import torch
from torch import nn
from torch.nn import functional as F


class AudioCNN(nn.Module):
    def __init__(self, classes, width=24):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(1, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.GELU()])
        # Preserve coarse frequency position: global frequency pooling loses tone identity.
        self.head = nn.Linear(width * 8, classes)

    def forward(self, features, lengths):
        """features [B,mel,time], lengths measured in spectrogram frames."""
        mask = torch.arange(features.size(-1), device=features.device)[None] < lengths[:, None]
        x = features[:, None] * mask[:, None, None]
        for layer in self.layers:
            x = layer(x) * mask[:, None, None]
        pooled = x.sum(-1) / lengths[:, None, None].clamp_min(1)
        return self.head(F.adaptive_avg_pool1d(pooled, 8).flatten(1))


def ctc_greedy(logits, lengths, blank=0):
    """[B,T,V] logits -> list of token lists; collapse repeats BEFORE removing blank."""
    results = []
    for ids, length in zip(logits.argmax(-1).tolist(), lengths.tolist()):
        sequence, previous = [], None
        for token in ids[:length]:
            if token != previous and token != blank:
                sequence.append(token)
            previous = token
        results.append(sequence)
    return results


def edit_distance(reference, hypothesis):
    """Works with words or characters; return integer error count."""
    previous = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, start=1):
        current = [i]
        for j, target in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (source != target)))
        previous = current
    return previous[-1]
