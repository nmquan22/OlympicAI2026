"""Paired augmentation preserves discrete class IDs in a semantic mask."""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as VF


class MaskPairs(Dataset):
    """CSV columns: image,mask. Paths relative to CSV. Masks store class IDs."""
    def __init__(self, manifest, training=False, size=(160, 160)):
        self.root = Path(manifest).parent
        with Path(manifest).open(encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        transforms = [v2.Resize(size, antialias=True)]
        if training:
            transforms.append(v2.RandomHorizontalFlip())
        self.geometry = v2.Compose(transforms)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            image = VF.to_image(source.convert("RGB"))
        with Image.open(self.root / row["mask"]) as source:
            mask = tv_tensors.Mask(torch.from_numpy(np.array(source, dtype=np.int64)))
        image, mask = self.geometry(image, mask)
        return image.float().div(255), mask.long()
