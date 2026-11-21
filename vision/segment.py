"""Train a two-scale U-Net on indexed PNG masks. Ignore 255."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .blocks.masks import MaskPairs
from .blocks.models import SmallUNet


def fixture(seed):
    rng = torch.Generator().manual_seed(seed)
    images = torch.rand(8, 3, 48, 48, generator=rng)
    masks = images[:, 0].gt(0.6).long()
    return TensorDataset(images, masks)


@torch.inference_mode()
def evaluate(model, loader, classes, device):
    model.eval()
    matrix = torch.zeros(classes, classes, dtype=torch.long)
    for images, labels in loader:
        pred = model(images.to(device)).argmax(1).cpu()
        keep = labels.ne(255)
        cells = labels[keep] * classes + pred[keep]
        matrix += torch.bincount(cells, minlength=classes**2).reshape(classes, classes)
    union = matrix.sum(0) + matrix.sum(1) - matrix.diag()
    present = union.gt(0)
    return (matrix.diag()[present] / union[present]).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/segment"))
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    torch.manual_seed(7)
    train, valid = (MaskPairs(args.train, True), MaskPairs(args.valid)) if args.train else (fixture(7), fixture(11))
    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=4)
    model = SmallUNet(args.classes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        model.train()
        for images, masks in train_loader:
            optimizer.zero_grad()
            logits = model(images.to(args.device))
            loss = F.cross_entropy(logits, masks.to(args.device), ignore_index=255, reduction="sum")
            (loss / max(1, masks.ne(255).sum().item())).backward()
            optimizer.step()
        print(f"epoch {epoch+1}, validation mIoU {evaluate(model, valid_loader, args.classes, args.device):.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "last.pt")


if __name__ == "__main__":
    main()
