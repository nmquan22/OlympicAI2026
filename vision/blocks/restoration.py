"""Lightweight residual restoration and aligned RGB-guided thermal SR."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualRestorer(nn.Module):
    def __init__(self, channels=3, width=32, depth=4):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(width, channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, image):
        return image + self.net(image)


class GuidedSR(nn.Module):
    """LR thermal [B,1,h,w] + registered HR RGB [B,3,H,W] -> HR thermal."""
    def __init__(self, width=32):
        super().__init__()
        self.thermal = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.GELU())
        self.rgb = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU())
        self.gate = nn.Conv2d(2 * width, width, 1)
        self.head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, low, rgb):
        base = F.interpolate(low, size=rgb.shape[-2:], mode="bicubic", align_corners=False)
        thermal, guide = self.thermal(base), self.rgb(rgb)
        weight = self.gate(torch.cat([thermal, guide], dim=1)).sigmoid()
        return base + self.head(thermal + weight * guide)


def charbonnier(predicted, target, epsilon=1e-3):
    return ((predicted - target).square() + epsilon**2).sqrt().mean()


def psnr(predicted, target, data_range=1.0):
    """Per-image mean PSNR. Caller chooses RGB/Y, border crop and data range."""
    mse = (predicted - target).square().flatten(1).mean(1)
    return (10 * torch.log10(data_range**2 / mse.clamp_min(1e-12))).mean()


@torch.inference_mode()
def tiled_restore(model, image, tile=128, halo=16):
    """Same-resolution model, batch size 1. Halo must cover its receptive radius."""
    if image.size(0) != 1 or tile < 1 or halo < 0:
        raise ValueError("Need batch size 1, positive tile and nonnegative halo")
    h, w = image.shape[-2:]
    result = torch.empty_like(image)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y1, x1 = max(y - halo, 0), max(x - halo, 0)
            y2, x2 = min(y + tile, h), min(x + tile, w)
            patch = model(image[..., y1:min(y2 + halo, h), x1:min(x2 + halo, w)])
            result[..., y:y2, x:x2] = patch[..., y-y1:y2-y1, x-x1:x2-x1]
    return result
