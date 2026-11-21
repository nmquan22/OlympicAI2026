"""Causal windows. Targets begin strictly after the last observed timestamp."""
import numpy as np
import torch
from torch import nn


def forecast_windows(values, context, horizon):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) < context + horizon:
        raise ValueError("Sequence shorter than context + horizon")
    origins = np.arange(context, len(values) - horizon + 1)
    x = np.stack([values[t - context:t] for t in origins])
    y = np.stack([values[t:t + horizon] for t in origins])
    return x, y, origins


def statistical_features(windows):
    """[N,T,C] -> [N,6C]; handles missing observations, not all-missing channels."""
    x = np.asarray(windows)
    return np.concatenate([np.nanmean(x, axis=1), np.nanstd(x, axis=1),
                           np.nanmin(x, axis=1), np.nanmax(x, axis=1),
                           x[:, -1], np.mean(np.isnan(x), axis=1)], axis=1)


def seasonal_naive(windows, horizon, period):
    if period > windows.shape[1] or period < 1:
        raise ValueError("Period must fit within the observed context")
    return windows[:, -period:][:, np.arange(horizon) % period]


class TemporalCNN(nn.Module):
    """Sequence classification. Input [B,T,C]; padded batches need masked pooling."""
    def __init__(self, channels, classes, width=32):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(width, width, 3, padding=1), nn.GELU())
        self.head = nn.Linear(width, classes)

    def forward(self, values, lengths):
        mask = torch.arange(values.size(1), device=values.device)[None] < lengths[:, None]
        features = (values * mask.unsqueeze(-1)).transpose(1, 2)
        for layer in self.features:
            features = layer(features) * mask[:, None]
        return self.head((features * mask[:, None]).sum(-1) / lengths[:, None].clamp_min(1))
