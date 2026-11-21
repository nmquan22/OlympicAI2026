"""IOAI 2027 extension: tabular Q-learning, DQN targets and REINFORCE."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def q_update(q, state, action, reward, next_state, terminated, lr=0.2, gamma=0.99):
    """A time-limit truncation is not a terminal MDP state: continue bootstrapping."""
    target = reward + gamma * (not terminated) * q[next_state].max()
    q[state, action] += lr * (target - q[state, action])


class QNetwork(nn.Module):
    def __init__(self, states, actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(states, hidden), nn.ReLU(), nn.Linear(hidden, actions))

    def forward(self, states):
        return self.net(states)


def dqn_loss(online, target, states, actions, rewards, next_states, terminated, gamma=0.99):
    with torch.no_grad():
        expected = rewards + gamma * (~terminated.bool()).float() * target(next_states).max(1).values
    predicted = online(states).gather(1, actions[:, None]).squeeze(1)
    return F.smooth_l1_loss(predicted, expected)


def reinforce_loss(log_probabilities, rewards, gamma=0.99):
    returns, total = [], 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.append(total)
    values = torch.tensor(returns[::-1], device=log_probabilities.device, dtype=log_probabilities.dtype)
    # A scalar baseline is optional; use an independently fitted baseline for tiny batches.
    return -(log_probabilities * values).sum()
