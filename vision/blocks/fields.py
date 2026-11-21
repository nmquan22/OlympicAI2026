"""Coordinate MLP teaching baseline; not the full stochastic IOAI Field solution."""
import torch
from torch import nn


class FourierField(nn.Module):
    def __init__(self, dimensions=2, width=32, frequencies=6, outputs=1):
        super().__init__()
        self.register_buffer("bands", 2.0 ** torch.arange(frequencies))
        inputs = dimensions * (1 + 2 * frequencies)
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width),
                                 nn.ReLU(), nn.Linear(width, outputs))

    def forward(self, coordinates):
        phase = 2 * torch.pi * coordinates[..., :, None] * self.bands
        encoded = torch.cat([coordinates, phase.sin().flatten(-2), phase.cos().flatten(-2)], dim=-1)
        return self.net(encoded)


def signed_log(values):
    return values.sign() * values.abs().log1p()


def signed_expm1(values):
    return values.sign() * values.abs().expm1()
