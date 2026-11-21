"""Train log-mel CNN on WAV CSV manifests or an offline synthetic frequency fixture."""
import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from common.runtime import seed_everything
from common.submission import write_predictions
from .blocks.features import read_wav, log_mel, spec_augment
from .blocks.models import AudioCNN


def read_manifest(path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [(row["id"], read_wav(path.parent / row["path"]), int(row["label"])) for row in rows]


def fixture(count, seed):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for i in range(count):
        label = i % 2
        t = torch.arange(4000 + 80 * (i % 3)) / 16000
        frequency = 300 + label * 1600 + 40 * torch.rand((), generator=generator)
        wave = 0.3 * torch.sin(2 * torch.pi * frequency * t) + 0.02 * torch.randn(len(t), generator=generator)
        result.append((f"{seed}-{i}", wave, label))
    return result


class Features:
    def __init__(self, rows, augment=False):
        self.rows = [(key, log_mel(wave), label) for key, wave, label in rows]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        key, mel, label = self.rows[index]
        mel = (mel - mel.mean()) / mel.std().clamp_min(1e-5)
        return key, spec_augment(mel) if self.augment else mel, label


def collate(rows):
    keys, features, labels = zip(*rows)
    lengths = torch.tensor([x.size(-1) for x in features])
    padded = torch.stack([F.pad(x, (0, int(lengths.max()) - x.size(-1))) for x in features])
    return keys, padded, lengths, torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/audio"))
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    seed_everything()
    train, valid = (read_manifest(args.train), read_manifest(args.valid)) if args.train else (fixture(24, 7), fixture(12, 8))
    if {r[0] for r in train} & {r[0] for r in valid}:
        parser.error("Train and validation recording IDs overlap")
    classes = 1 + max(row[2] for row in train)
    training = DataLoader(Features(train, augment=True), batch_size=8, shuffle=True, collate_fn=collate)
    validation = DataLoader(Features(valid), batch_size=8, collate_fn=collate)
    model = AudioCNN(classes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for _ in range(args.epochs):
        model.train()
        for _, features, lengths, labels in training:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(features.to(args.device), lengths.to(args.device)), labels.to(args.device))
            loss.backward()
            optimizer.step()
    model.eval()
    ids, targets, predictions = [], [], []
    with torch.inference_mode():
        for keys, features, lengths, labels in validation:
            ids.extend(keys)
            targets.extend(labels.tolist())
            predictions.extend(model(features.to(args.device), lengths.to(args.device)).argmax(1).cpu().tolist())
    write_predictions(args.out / "predictions.csv", ids, predictions)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "sample_rate": 16000}, args.out / "model.pt")
    print({"data": "manifest" if args.train else "synthetic_fixture", "macro_f1": float(f1_score(targets, predictions, average="macro"))})


if __name__ == "__main__":
    main()
