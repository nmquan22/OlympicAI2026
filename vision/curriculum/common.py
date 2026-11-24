import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, swin_t, Swin_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor


def backbone(pretrained=False):
    m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    return create_feature_extractor(
        m, {"layer2": "fine", "layer3": "mid", "layer4": "coarse"}
    )


def local_backbone(path):
    m = resnet18(weights=None)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return create_feature_extractor(
        m, {"layer2": "fine", "layer3": "mid", "layer4": "coarse"}
    )


def tokens(x):
    return x.flatten(2).transpose(1, 2)


def pool(x):
    return x.mean((-2, -1))


class Encoder(nn.Module):

    def __init__(self, d=128):
        super().__init__()
        self.net = create_feature_extractor(resnet18(weights=None), {"layer2": "fine"})
        self.proj = nn.Conv2d(128, d, 1)

    def forward(self, x):
        return self.proj(self.net(x)["fine"])


class Cross(nn.Module):

    def __init__(self, d=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, k):
        return self.norm(q + self.attn(q, k, k, need_weights=False)[0])


def transformer(d=128, layers=2):
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d, 4, d * 4, batch_first=True, norm_first=True),
        layers,
    )


def dice_loss(logits, y):
    p = logits.sigmoid().flatten(1)
    y = y.float().flatten(1)
    return 1 - ((2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)).mean()


def train_step(model, optimizer, loss_fn, batch, device="cuda"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = [x.to(device) for x in batch]
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda" and torch.cuda.is_bf16_supported(),
    ):
        loss = loss_fn(model, *batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach().item()
