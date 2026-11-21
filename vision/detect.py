"""Train a small Faster R-CNN and export validation detections in original COCO coordinates."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from .blocks.data import CocoBoxes, collate_boxes, detection_transforms
from .blocks.models import tiny_detector


def write_fixture(root, split, offset):
    root.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for index in range(6):
        x, y = 6 + index * 5 + offset, 10 + 3*index
        image = Image.new("RGB", (112, 80), "navy")
        name = f"{split}-{index}.png"
        if index:
            ImageDraw.Draw(image).rectangle((x, y, x+23, y+17), fill="orange")
            annotations.append(dict(id=index, image_id=index, category_id=7, bbox=[x, y, 24, 18], area=24*18, iscrowd=0))
        image.save(root / name)
        images.append(dict(id=index, file_name=name, width=112, height=80))
    path = root / f"{split}.json"
    path.write_text(json.dumps(dict(info={}, images=images, annotations=annotations, categories=[dict(id=7, name="box")])))
    return path


@torch.inference_mode()
def predict(model, dataset, device):
    model.eval()
    categories = {v: k for k, v in dataset.category_ids.items()}
    rows = []
    for index, record in enumerate(dataset.images):
        image, _ = dataset[index]
        pred = model([image.to(device)])[0]
        height, width = image.shape[-2:]
        scale = pred["boxes"].new_tensor([record["width"]/width, record["height"]/height] * 2)
        original = box_convert(pred["boxes"] * scale, "xyxy", "xywh")
        for box, label, score in zip(original.tolist(), pred["labels"].tolist(), pred["scores"].tolist()):
            rows.append(dict(image_id=record["id"], category_id=categories[label], bbox=box, score=score))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/detection"))
    args = parser.parse_args()
    torch.manual_seed(7)
    paths = (args.images, args.train, args.valid)
    if any(paths) and not all(paths):
        parser.error("Supply --images, --train and --valid together")
    root = args.images or args.out / "fixture"
    train_path = args.train or write_fixture(root, "train", 0)
    valid_path = args.valid or write_fixture(root, "valid", 2)
    train = CocoBoxes(root, train_path, detection_transforms(True))
    valid = CocoBoxes(root, valid_path, detection_transforms(False), train.category_ids)
    loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_boxes)
    model = tiny_detector(len(train.category_ids)+1).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for images, targets in loader:
            images = [image.to(args.device) for image in images]
            targets = [{k: target[k].to(args.device) for k in ("boxes", "labels")} for target in targets]
            optimizer.zero_grad()
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1}, last batch loss {loss.item():.4f}")
    torch.save({"model": model.state_dict(), "category_ids": train.category_ids}, args.out / "last.pt")
    rows = predict(model, valid, args.device)
    (args.out / "detections.json").write_text(json.dumps(rows))
    print(f"{len(rows)} detections for {len(valid)} validation images")


if __name__ == "__main__":
    main()
