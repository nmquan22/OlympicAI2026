"""Train same-resolution residual restoration on aligned CSV pairs or an offline fixture."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from common.runtime import seed_everything
from .blocks.restoration import ResidualRestorer, charbonnier, psnr


def read_pairs(path, scale):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    image = lambda p: torch.from_numpy(np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    pairs = []
    for row in rows:
        low, high = image(path.parent / row["degraded"]), image(path.parent / row["target"])
        low = F.interpolate(low, scale_factor=scale, mode="bicubic", align_corners=False)
        if low.shape != high.shape:
            raise ValueError("Degraded dimensions * scale must match target dimensions")
        pairs.append((row["id"], low, high))
    return pairs


def fixture(seed, count):
    generator = torch.Generator().manual_seed(seed)
    pairs = []
    for i in range(count):
        clean = F.interpolate(torch.rand(1, 3, 8, 8, generator=generator), size=(32, 32), mode="bilinear", align_corners=False)
        noisy = (clean + 0.08 * torch.randn(clean.shape, generator=generator)).clamp(0, 1)
        pairs.append((f"{seed}-{i}", noisy, clean))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/restoration"))
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    if args.scale < 1:
        parser.error("Scale must be positive")
    seed_everything()
    train, valid = (read_pairs(args.train, args.scale), read_pairs(args.valid, args.scale)) if args.train else (fixture(7, 12), fixture(8, 4))
    if {r[0] for r in train} & {r[0] for r in valid}:
        parser.error("Train and validation image IDs overlap")
    model = ResidualRestorer(width=16, depth=3).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(args.epochs):
        model.train()
        for _, low, high in train:
            optimizer.zero_grad()
            loss = charbonnier(model(low.to(args.device)), high.to(args.device))
            loss.backward()
            optimizer.step()
    model.eval()
    before, after = [], []
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, (_, low, high) in enumerate(valid):
            prediction = model(low.to(args.device)).clamp(0, 1).cpu()
            before.append(float(psnr(low.clamp(0, 1), high)))
            after.append(float(psnr(prediction, high)))
            pixels = (prediction[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(args.out / f"prediction_{index:04d}.png")
    torch.save({"state_dict": model.state_dict(), "width": 16, "depth": 3, "scale": args.scale}, args.out / "model.pt")
    print({"data": "csv" if args.train else "synthetic_fixture", "input_psnr": float(np.mean(before)), "output_psnr": float(np.mean(after))})


if __name__ == "__main__":
    main()
