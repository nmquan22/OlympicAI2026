"""Train on real JSONL or run the self-contained smoke example."""
import argparse
import math
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .blocks.bpe import BPE
from .blocks.data import TextDataset, collate_text, read_jsonl
from .blocks.transformers import DecoderLM, EncoderClassifier


def fixture():
    train = [
        {"text": text, "label": label} for text, label in [
            ("giảng viên giải thích rõ", 1), ("bài tập dễ hiểu", 1),
            ("tiết học rất hay", 1), ("bài tập hơi khó", 0),
            ("âm thanh quá nhỏ", 0), ("lịch học khó theo", 0),
        ]
    ]
    return train, [{"text": "bài học rất hay", "label": 1}, {"text": "bài quá khó", "label": 0}]


def encoder_step(model, ids, labels):
    return F.cross_entropy(model(ids), labels)


def lm_step(model, ids, labels):
    return model.loss(ids)


def evaluate(model, loader, mode, device):
    model.eval()
    labels, predictions = [], []
    total_nll, tokens = 0.0, 0
    with torch.inference_mode():
        for ids, target in loader:
            ids = ids.to(device)
            if mode == "lm":
                count = ids[:, 1:].ne(0).sum().item()
                total_nll += model.loss(ids).item() * count
                tokens += count
            else:
                labels.extend(target.tolist())
                predictions.extend(model(ids).argmax(1).tolist())
    if mode == "lm":
        return {"perplexity": math.exp(total_nll / tokens)}
    return {"macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--mode", choices=["encoder", "lm"], default="encoder")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/nlp"))
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    torch.manual_seed(7)
    train, valid = (read_jsonl(args.train), read_jsonl(args.valid)) if args.train else fixture()
    # Unlabelled language-model corpora use the same batch layout.
    if args.mode == "lm":
        train, valid = ([dict(row, label=0) for row in rows] for rows in (train, valid))
    tokenizer = BPE.train((row["text"] for row in train), num_merges=100)
    datasets = [TextDataset(rows, tokenizer) for rows in (train, valid)]
    training = DataLoader(datasets[0], batch_size=16, shuffle=True, collate_fn=collate_text)
    validation = DataLoader(datasets[1], batch_size=32, collate_fn=collate_text)
    vocab_size = len(tokenizer.vocab)
    if args.mode == "lm":
        model, step = DecoderLM(vocab_size), lm_step
    else:
        classes = 1 + max(row["label"] for row in train)
        model, step = EncoderClassifier(vocab_size, classes), encoder_step
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out / "bpe.json")
    for epoch in range(args.epochs):
        model.train()
        for ids, labels in training:
            optimizer.zero_grad()
            loss = step(model, ids.to(args.device), labels.to(args.device))
            loss.backward()
            optimizer.step()
        print(epoch + 1, evaluate(model, validation, args.mode, args.device))
    torch.save({"mode": args.mode, "state_dict": model.state_dict(), "vocab_size": vocab_size,
                "output_size": model.head.out_features}, args.out / "last.pt")


if __name__ == "__main__":
    main()
