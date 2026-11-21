"""Canonical box format is float XYXY in continuous image-edge coordinates."""
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as VF


def read_voc(xml_path, class_ids):
    root = ET.parse(xml_path).getroot()
    objects = root.findall("object")
    boxes, labels, difficult = [], [], []
    for obj in objects:
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([x1 - 1, y1 - 1, x2, y2])
        labels.append(class_ids[obj.findtext("name")])
        difficult.append(int(obj.findtext("difficult", "0")))
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.long), difficult


def read_yolo(txt_path, size):
    """YOLO labels start at 0. Detector reserves class 0 for background."""
    width, height = size
    boxes, labels = [], []
    for line in Path(txt_path).read_text().splitlines():
        if line.strip():
            label, cx, cy, w, h = map(float, line.split())
            boxes.append([(cx - w/2)*width, (cy - h/2)*height,
                          (cx + w/2)*width, (cy + h/2)*height])
            labels.append(int(label) + 1)
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.long)


def detection_transforms(training, size=(160, 160)):
    steps = [v2.Resize(size, antialias=True)]
    if training:
        steps.extend([v2.RandomHorizontalFlip(0.5), v2.ColorJitter(brightness=0.15, contrast=0.15)])
    steps.extend([v2.ClampBoundingBoxes(), v2.SanitizeBoundingBoxes(), v2.ToDtype(torch.float32, scale=True)])
    return v2.Compose(steps)


class CocoBoxes(Dataset):
    """Ignore crowd annotations for training. Keep original JSON for official COCO eval."""
    def __init__(self, root, annotations, transforms, category_ids=None):
        self.root = Path(root)
        data = json.loads(Path(annotations).read_text())
        self.images = data["images"]
        ids = sorted(c["id"] for c in data["categories"])
        self.category_ids = dict(category_ids) if category_ids is not None else {v: i+1 for i, v in enumerate(ids)}
        self.annotations = defaultdict(list)
        for ann in data["annotations"]:
            if not ann.get("iscrowd", 0):
                self.annotations[ann["image_id"]].append(ann)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        with Image.open(self.root / record["file_name"]) as source:
            image = VF.to_image(source.convert("RGB"))
        entries = self.annotations[record["id"]]
        xywh = torch.tensor([a["bbox"] for a in entries], dtype=torch.float32).reshape(-1, 4)
        boxes = tv_tensors.BoundingBoxes(xywh, format="XYWH", canvas_size=image.shape[-2:])
        target = {
            "boxes": VF.convert_bounding_box_format(boxes, new_format="XYXY"),
            "labels": torch.tensor([self.category_ids[a["category_id"]] for a in entries], dtype=torch.long),
            "image_id": record["id"],
        }
        return self.transforms(image, target)


def collate_boxes(examples):
    images, targets = zip(*examples)
    return list(images), list(targets)
