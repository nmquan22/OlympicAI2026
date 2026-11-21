"""Pixel prototype baseline on Omniglot evaluation alphabets, fixed episodes."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .blocks.metrics import prototypes


def character_images(root):
    features = []
    for folder in sorted(root.glob("*/*")):
        images = []
        for path in sorted(folder.glob("*.png")):
            with Image.open(path) as source:
                gray = source.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
            images.append((1 - np.asarray(gray, dtype="float32")/255).ravel())
        if images:
            features.append(np.stack(images))
    return features


def episode(features, rng, way, shot):
    chosen = rng.choice(len(features), size=way, replace=False)
    support, query = [], []
    for char in chosen:
        images = features[char]
        sel = rng.choice(len(images), size=shot+1, replace=False)
        support.extend(images[sel[:shot]])
        query.append(images[sel[-1]])
    labels = np.arange(way)
    predicted = prototypes(np.stack(support), labels.repeat(shot), np.stack(query))
    return (predicted == labels).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted images_evaluation folder")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--way", type=int, default=20)
    parser.add_argument("--shot", type=int, default=1)
    args = parser.parse_args()
    features = character_images(args.root)
    rng = np.random.default_rng(7)
    scores = [episode(features, rng, args.way, args.shot) for _ in range(args.episodes)]
    print(f"{len(features)} characters, {np.mean(scores):.4f} accuracy, "
          f"episode standard error {np.std(scores, ddof=1)/np.sqrt(len(scores)):.4f}")


if __name__ == "__main__":
    main()
