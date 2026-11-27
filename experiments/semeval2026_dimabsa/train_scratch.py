"""Train small random-initialized DimASR encoders; no pretrained tokenizer/model."""
import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from cpu_baselines import export_predictions, metrics
from prepare_data import SOURCE_COMMIT, task1_splits


@dataclass
class Config:
    run: str = "R01"
    encoder: str = "gru"
    pooling: str = "meanmax"
    loss: str = "mse"
    seed: int = 42
    vocab: int = 3000
    max_length: int = 160
    width: int = 128
    layers: int = 2
    dropout: float = 0.25
    batch_size: int = 32
    epochs: int = 40
    patience: int = 6
    lr: float = 0.002
    seconds: int = 1200


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_tokenizer(examples, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((e["text"].lower() for e in examples), trainer=trainer)
    return tokenizer


class AspectDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.items = []
        for example in examples:
            text, aspect = example["text"].lower(), example["aspect"].lower()
            encoding = tokenizer.encode(text)
            aspect_ids = tokenizer.encode(aspect).ids[:24]
            budget = max_length - len(aspect_ids) - 3
            aspect_start = text.find(aspect)
            positions = [i for i, (start, end) in enumerate(encoding.offsets)
                         if start < aspect_start + len(aspect) and end > aspect_start and aspect_start >= 0]
            center = positions[0] if positions else 0
            start = max(0, min(center - budget // 2, len(encoding.ids) - budget))
            text_ids = encoding.ids[start:start + budget]
            ids = [2] + text_ids + [3] + aspect_ids + [3]
            context = [False] + [True] * len(text_ids) + [False] * (len(aspect_ids) + 2)
            query = [False] * (len(text_ids) + 2) + [True] * len(aspect_ids) + [False]
            segment = [0] * (len(text_ids) + 2) + [1] * (len(aspect_ids) + 1)
            self.items.append((ids, context, query, segment, example["target"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items):
    size = max(len(item[0]) for item in items)
    columns = []
    for k, dtype in enumerate((torch.long, torch.bool, torch.bool, torch.long)):
        columns.append(torch.tensor([item[k] + [0] * (size - len(item[k])) for item in items], dtype=dtype))
    return (*columns, torch.tensor([item[4] for item in items], dtype=torch.float32))


def masked_mean(hidden, mask):
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)


class ScratchRegressor(nn.Module):
    def __init__(self, cfg, vocab_size, target_mean):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.width, padding_idx=0)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.encoder == "gru":
            self.encoder = nn.GRU(cfg.width, cfg.width // 2, num_layers=cfg.layers,
                                  bidirectional=True, batch_first=True, dropout=cfg.dropout)
        else:
            self.position = nn.Embedding(cfg.max_length, cfg.width)
            self.segment = nn.Embedding(2, cfg.width)
            block = nn.TransformerEncoderLayer(cfg.width, 4, cfg.width * 4, cfg.dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(block, cfg.layers, norm=nn.LayerNorm(cfg.width),
                                                 enable_nested_tensor=False)
        pooled_width = cfg.width * (4 if cfg.pooling == "attention" else 3)
        self.head = nn.Sequential(nn.Linear(pooled_width, cfg.width), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(cfg.width, 2))
        self.log_variance = nn.Parameter(torch.zeros(2), requires_grad=cfg.loss == "uncertainty")
        prior = torch.as_tensor((target_mean - 1.0) / 8.0).clamp(0.01, 0.99)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.logit(prior))

    def forward(self, ids, context, query, segment):
        hidden = self.dropout(self.embedding(ids))
        padding = ids.eq(0)
        if self.cfg.encoder == "gru":
            packed = pack_padded_sequence(hidden, (~padding).sum(1).cpu(), batch_first=True, enforce_sorted=False)
            packed, _ = self.encoder(packed)
            hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=ids.shape[1])
        else:
            pos = torch.arange(ids.shape[1], device=ids.device)
            hidden = self.encoder(hidden + self.position(pos)[None] + self.segment(segment), src_key_padding_mask=padding)
        mean = masked_mean(hidden, context)
        maximum = hidden.masked_fill(~context.unsqueeze(-1), -1e4).amax(1)
        aspect = masked_mean(hidden, query)
        pooled = [mean, maximum, aspect]
        if self.cfg.pooling == "attention":
            logits = torch.einsum("bld,bd->bl", hidden, aspect) / math.sqrt(hidden.shape[-1])
            attention = logits.masked_fill(~context, -1e4).softmax(-1)
            pooled.append(torch.einsum("bl,bld->bd", attention, hidden))
        return 1.0 + 8.0 * torch.sigmoid(self.head(torch.cat(pooled, dim=-1)))

    def objective(self, pred, target):
        per_dimension = ((pred - target) ** 2).mean(0)
        if self.cfg.loss == "uncertainty":
            return (0.5 * (torch.exp(-self.log_variance) * per_dimension + self.log_variance)).sum()
        return per_dimension.mean()


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        predictions.append(model(*(tensor.to(device) for tensor in batch[:4])).cpu().numpy())
    return np.round(np.concatenate(predictions), 2)


def train(cfg, data, output):
    seed_everything(cfg.seed)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    train_examples, dev_examples, dev_rows, removed = task1_splits(data)
    tokenizer = fit_tokenizer(train_examples, cfg.vocab)
    tokenizer.save(str(output / "tokenizer.json"))
    make_dataset = lambda examples: AspectDataset(examples, tokenizer, cfg.max_length)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(make_dataset(train_examples), batch_size=cfg.batch_size, shuffle=True,
                              generator=generator, collate_fn=collate, num_workers=0)
    dev_loader = DataLoader(make_dataset(dev_examples), batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = np.array([e["target"] for e in train_examples])
    y_dev = np.array([e["target"] for e in dev_examples])
    model = ScratchRegressor(cfg, tokenizer.get_vocab_size(), targets.mean(0)).to(device)
    weights = [p for name, p in model.named_parameters() if name != "log_variance"]
    optimizer = torch.optim.AdamW([{"params": weights, "lr": cfg.lr, "weight_decay": 0.01},
                                   {"params": [model.log_variance], "lr": 0.01, "weight_decay": 0.0}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    best, stale, best_epoch, best_state = float("inf"), 0, 0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            pred = model(*batch[:4])
            loss = model.objective(pred, batch[4])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        scores = metrics(y_dev, predict(model, dev_loader, device))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.monotonic() - start,
                 "log_variance": model.log_variance.detach().cpu().tolist(), **scores}
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps({"run": cfg.run, **entry}), flush=True)
        score = scores["official_RMSE_VA"]
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg.patience or time.monotonic() - start >= cfg.seconds:
            break
    model.load_state_dict(best_state)
    pred = predict(model, dev_loader, device)
    export_predictions(dev_rows, dev_examples, pred, output / "dev_predictions.jsonl")
    torch.save({"config": asdict(cfg), "state_dict": best_state}, output / "best.pt")
    report = {"status": "completed", "config": asdict(cfg), "pretrained_weights": False,
              "tokenizer_fit": "clean train text only", "source_commit": SOURCE_COMMIT,
              "train_aspects": len(train_examples), "dev_aspects": len(dev_examples),
              "overlapping_train_rows_removed": removed, "best_epoch": best_epoch,
              "parameters": sum(p.numel() for p in model.parameters()), "seconds": time.monotonic() - start,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
              "torch_version": torch.__version__, "test_evaluated": False,
              "tokenizer_sha256": hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest(),
              **metrics(y_dev, pred)}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", default="R01")
    parser.add_argument("--encoder", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--pooling", choices=["meanmax", "attention"], default="meanmax")
    parser.add_argument("--loss", choices=["mse", "uncertainty"], default="mse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    config_keys = ("run", "encoder", "pooling", "loss", "seed", "epochs", "seconds")
    cfg = Config(**{key: getattr(args, key) for key in config_keys})
    if (args.out / "epochs.jsonl").exists():
        raise FileExistsError("Use a new output directory for an independent experiment.")
    train(cfg, args.data, args.out)


if __name__ == "__main__":
    main()
