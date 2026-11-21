"""Small batch-first Transformers. PAD=0, BOS=2, EOS=3 in the reference BPE."""
import torch
from torch import nn
from torch.nn import functional as F


class TokenPositions(nn.Module):
    def __init__(self, vocab_size, width, max_length):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, width, padding_idx=0)
        self.positions = nn.Embedding(max_length, width)

    def forward(self, ids):
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.tokens(ids) + self.positions(positions)


def blocks(width, heads, depth):
    layer = nn.TransformerEncoderLayer(
        width, heads, dim_feedforward=4 * width, batch_first=True, activation="gelu"
    )
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, classes, width=64, heads=4, depth=2, max_length=128):
        super().__init__()
        self.embed = TokenPositions(vocab_size, width, max_length)
        self.encoder = blocks(width, heads, depth)
        self.head = nn.Linear(width, classes)

    def forward(self, ids):
        valid = ids.ne(0)
        features = self.encoder(self.embed(ids), src_key_padding_mask=~valid)
        pooled = (features * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)


class DecoderLM(nn.Module):
    """Decoder-only: masked self-attention, no cross-attention or encoder memory."""
    def __init__(self, vocab_size, width=64, heads=4, depth=2, max_length=128):
        super().__init__()
        self.embed = TokenPositions(vocab_size, width, max_length)
        self.decoder = blocks(width, heads, depth)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.max_length = max_length

    def forward(self, ids):
        length = ids.size(1)
        future = torch.ones((length, length), dtype=torch.bool, device=ids.device).triu(1)
        states = self.decoder(self.embed(ids), mask=future, src_key_padding_mask=ids.eq(0))
        return self.head(states)

    def loss(self, ids):
        logits = self(ids[:, :-1])
        return F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], ignore_index=0)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=20, eos_id=3):
        """Greedy generation for one unpadded prompt of shape [1, T], in eval mode."""
        self.eval()
        for _ in range(max_new_tokens):
            next_id = self(prompt[:, -self.max_length:])[:, -1].argmax(-1, keepdim=True)
            prompt = torch.cat([prompt, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return prompt
