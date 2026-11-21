"""CPU patch-memory baseline. The local category folder follows the MVTec layout."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from .blocks.anomaly import PatchBank


def patch_moments(image, patch=8):
    pixels = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR), dtype="float32") / 255
    grid = pixels.reshape(128//patch, patch, 128//patch, patch, 3)
    return np.concatenate((grid.mean((1, 3)), grid.std((1, 3))), axis=-1)


def read_features(paths):
    rows = []
    for path in paths:
        with Image.open(path) as image:
            rows.append(patch_moments(image))
    return np.stack(rows)


def images(folder):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)


def local_category(root):
    normal = images(root / "train" / "good")
    test = images(root / "test")
    labels = np.array([p.relative_to(root / "test").parts[0] != "good" for p in test])
    return read_features(normal), read_features(test), labels, [str(p.relative_to(root)) for p in test]


def fixture():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(12, 8, 8, 6))
    test = rng.normal(size=(10, 8, 8, 6))
    test[5:, 2:4, 4:6] += 4
    return train, test, np.arange(10) >= 5, [f"demo-{n}" for n in range(10)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/anomaly"))
    args = parser.parse_args()
    normal, test, labels, names = local_category(args.category) if args.category else fixture()
    bank = PatchBank().fit(normal)
    scores, maps = bank.score(test)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "score"])
        writer.writerows(zip(names, scores))
    np.save(args.out / "heatmaps.npy", maps)
    print(f"image AUROC {roc_auc_score(labels, scores):.4f}")


if __name__ == "__main__":
    main()
