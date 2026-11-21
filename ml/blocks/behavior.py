"""Supervised grid/instruction policy for Robot Chasing style snapshots."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class InstructionPolicy(nn.Module):
    def __init__(self, vocabulary, width=12, hidden=64, grid_size=8, actions=6):
        super().__init__()
        self.objects = nn.Embedding(12, width)
        self.colors = nn.Embedding(6, width)
        self.robots = nn.Embedding(6, width)
        self.directions = nn.Embedding(4, width)
        self.words = nn.Embedding(vocabulary, width, padding_idx=0)
        self.text = nn.GRU(width, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear((grid_size**2 * 2 + 5) * width, hidden), nn.GELU(), nn.Linear(hidden, actions))

    def forward(self, grid, robot_ids, directions, carrying, missions, lengths):
        """grid [B,8,8,2], carrying [B,2] with object=0 for none, padded token IDs.

        IDs follow IOAI 2026 statement, not RGB intensities. Lengths must be >=1.
        Outputs absolute movement/pickup/drop action logits, not path coordinates.
        """
        objects = self.objects(grid[..., 0])
        meaningful_color = ~((grid[..., 0] == 1) | (grid[..., 0] == 2))
        colors = self.colors(grid[..., 1]) * meaningful_color[..., None]
        spatial = torch.cat([objects, colors], dim=-1).flatten(1)
        packed = pack_padded_sequence(self.words(missions), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, text = self.text(packed)
        has_object = carrying[:, 0:1].ne(0)
        features = torch.cat([spatial, self.robots(robot_ids), self.directions(directions), text[-1],
                              self.objects(carrying[:, 0]) * has_object,
                              self.colors(carrying[:, 1]) * has_object], dim=-1)
        return self.head(features)
