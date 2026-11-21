"""Train/evaluate a multi-positive retrieval head on paired cached embeddings."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .blocks.fusion import DualEncoder, matching_loss
from .blocks.metrics import recall_at_k


def read_pairs(path):
    with np.load(path, allow_pickle=False) as data:
        return data["image"].astype("float32"), data["text"].astype("float32"), data["image_id"].astype(str)


def fixture(rng, count, prefix, image_projection, text_projection):
    hidden = rng.normal(size=(count, image_projection.shape[0])).repeat(2, axis=0)
    # Two independent captions per image and identical image features for the same ID.
    image = hidden @ image_projection
    text = hidden @ text_projection + rng.normal(0, 0.1, (2*count, text_projection.shape[1]))
    ids = np.asarray([f"{prefix}-{i}" for i in range(count)]).repeat(2)
    return image.astype("float32"), text.astype("float32"), ids


@torch.inference_mode()
def evaluate(model, valid, device):
    image, text, ids = valid
    model.eval()
    _, unique = np.unique(ids, return_index=True)
    # The order of unique indices defines both gallery vectors and gallery IDs.
    gallery, captions = model(torch.from_numpy(image[unique]).to(device), torch.from_numpy(text).to(device))
    similarities = (captions @ gallery.T).cpu().numpy()
    result = {}
    for k in (1, 5, 10):
        result[f"t2i_R@{k}"] = recall_at_k(similarities, ids, ids[unique], k)
        result[f"i2t_R@{k}"] = recall_at_k(similarities.T, ids[unique], ids, k)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("runs/retrieval"))
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    torch.manual_seed(7)
    if args.train:
        train, valid = read_pairs(args.train), read_pairs(args.valid)
    else:
        rng = np.random.default_rng(7)
        maps = rng.normal(size=(8, 16)), rng.normal(size=(8, 12))
        train, valid = fixture(rng, 64, "train", *maps), fixture(rng, 16, "valid", *maps)
    image, text, ids = train
    if np.intersect1d(ids, valid[2]).size:
        raise ValueError("Train and validation share image IDs")
    _, group = np.unique(ids, return_inverse=True)
    loader = DataLoader(TensorDataset(*map(torch.from_numpy, (image, text, group))), batch_size=32, shuffle=True)
    model = DualEncoder(image.shape[1], text.shape[1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    for epoch in range(args.epochs):
        model.train()
        for images, captions, groups in loader:
            opt.zero_grad()
            a, b = model(images.to(args.device), captions.to(args.device))
            loss = matching_loss(a, b, groups.to(args.device))
            loss.backward()
            opt.step()
        print(epoch + 1, evaluate(model, valid, args.device))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "image_dim": image.shape[1], "text_dim": text.shape[1]},
               args.out / "last.pt")


if __name__ == "__main__":
    main()
