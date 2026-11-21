"""Canonical local format: one JSON object per line; text and integer label."""
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.rows = records
        self.ids = [torch.tensor(tokenizer.encode(row["text"], max_length)) for row in records]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.ids[index], self.rows[index]["label"]


def collate_text(samples):
    tokens, labels = zip(*samples)
    return pad_sequence(tokens, batch_first=True, padding_value=0), torch.tensor(labels)
