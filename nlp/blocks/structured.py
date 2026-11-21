"""Useful when output is a span, set of labels, or schema-constrained structure."""
import numpy as np
import torch
from torch.nn import functional as F


def masked_token_loss(logits, labels, ignore_index=-100):
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=ignore_index)


def span_char_offsets(offset_mapping, token_start, token_end):
    """Token end is inclusive; returned original-text character end is exclusive."""
    return int(offset_mapping[token_start][0]), int(offset_mapping[token_end][1])


def best_span(start_logits, end_logits, valid_tokens, max_length=30):
    """One sequence, [L] scores; exclude padding and question tokens in valid_tokens."""
    length = len(start_logits)
    i, j = np.indices((length, length))
    valid = (j >= i) & (j - i < max_length) & valid_tokens[:, None] & valid_tokens[None, :]
    if not valid.any():
        raise ValueError("No valid answer span")
    scores = start_logits[:, None] + end_logits[None, :]
    return tuple(map(int, np.unravel_index(np.where(valid, scores, -np.inf).argmax(), scores.shape)))
