import torch
from torch import nn
from torch.nn import functional as F


class DualEncoder(nn.Module):
    """Inputs are cached vectors from local backbones, not pixels or token IDs."""
    def __init__(self, image_dim, text_dim, width=64):
        super().__init__()
        self.image = nn.Linear(image_dim, width)
        self.text = nn.Linear(text_dim, width)

    def forward(self, image, text):
        return F.normalize(self.image(image), dim=-1), F.normalize(self.text(text), dim=-1)


def matching_loss(image, text, group_ids, temperature=0.07):
    logits = image @ text.T / temperature
    positive = group_ids[:, None].eq(group_ids[None, :])
    weights = positive.float() / positive.sum(1, keepdim=True)
    return -(weights * (logits.log_softmax(1) + logits.T.log_softmax(1))).sum(1).mean() / 2


class SpatialQuestionHead(nn.Module):
    """One question vector attends to N spatial features, e.g. 7x7 patches."""
    def __init__(self, region_dim, question_dim, width, answers):
        super().__init__()
        self.keys = nn.Linear(region_dim, width)
        self.query = nn.Linear(question_dim, width)
        self.head = nn.Linear(region_dim + question_dim, answers)

    def forward(self, regions, question):
        keys, query = self.keys(regions), self.query(question)
        scores = (keys * query[:, None]).sum(-1) / keys.size(-1)**0.5
        pooled = (regions * scores.softmax(1).unsqueeze(-1)).sum(1)
        return self.head(torch.cat([pooled, question], dim=-1))
