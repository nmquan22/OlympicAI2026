"""Utilities for controlled, idea-driven AICC NLP experiments."""
import ast, csv, hashlib, importlib.metadata, json, math, os, random, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SEED = 42
OUT = Path("/kaggle/working")
DEVICE = torch.device("cuda:0")


def start_run(task, run_id, config):
    assert torch.cuda.is_available(), "GPU T4 is required; do not silently fall back to CPU"
    name = torch.cuda.get_device_name(0)
    assert "T4" in name, "Expected T4, received " + name
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    info = dict(task=task, run_id=f"R{run_id:02d}", config=config, seed=SEED,
                device=name, gpu_count=torch.cuda.device_count(), started=time.time(),
                source_commit=globals().get("SOURCE_COMMIT"),
                versions={p: importlib.metadata.version(p) for p in
                          ["torch", "numpy", "pandas", "scikit-learn", "tokenizers"]})
    print("RUN_START " + json.dumps(info), flush=True)
    return info


def find_data(slug):
    for root in (Path("/kaggle/input/competitions")/slug, Path("/kaggle/input")/slug):
        if root.is_dir():
            return root
    raise FileNotFoundError("Attach competition data: " + slug)


def make_bpe(texts, size=2048):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=size, min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    return tokenizer


def encode_texts(tokenizer, texts, max_length):
    values = np.zeros((len(texts), max_length), dtype=np.int64)
    for i, encoded in enumerate(tokenizer.encode_batch(texts)):
        ids = encoded.ids[:max_length] or [1]
        values[i, :len(ids)] = ids
    return values


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish_run(info, validation, submission, checks, history, artifacts=None, limitations=None):
    assert checks.get("ids_unique") is True and checks.get("schema_valid") is True
    assert checks.get("row_count", 0) > 0
    submission.to_csv(OUT/"submission.csv", index=False)
    info.update(status="completed", elapsed_seconds=time.time()-info.pop("started"),
                validation=validation, submission_checks=checks, history=history,
                peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                artifacts=artifacts or {}, limitations=limitations or [],
                leaderboard_public=None, leaderboard_private=None)
    (OUT/"run_report.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("NLP_RUN_RESULT " + json.dumps(info), flush=True)
    print("NLP_RUN_PASSED", flush=True)
    return info


class PositionEncoder(nn.Module):
    def __init__(self, vocab, dim=96, layers=2, max_length=384, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.position = nn.Embedding(max_length, dim)
        block = nn.TransformerEncoderLayer(dim, 4, dim*3, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids):
        mask = ids.eq(0)
        x = self.embedding(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x), mask


def pools(hidden, mask):
    valid = (~mask).unsqueeze(-1)
    mean = (hidden*valid).sum(1)/valid.sum(1).clamp(min=1)
    maximum = hidden.masked_fill(~valid, -1e4).amax(1)
    return torch.cat([mean, maximum], -1)

