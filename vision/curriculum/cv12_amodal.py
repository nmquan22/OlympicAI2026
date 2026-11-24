from .common import F, dice_loss, nn, torch


class Amodal(nn.Module):

    def __init__(self, d=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, d, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(d * 2, d * 2, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(d * 3, 1, 1)

    def forward(self, x, visible):
        f = self.enc(torch.cat([x, visible], 1))
        u = F.interpolate(
            self.down(f), f.shape[-2:], mode="bilinear", align_corners=False
        )
        hidden = self.out(torch.cat([f, u], 1))
        return hidden


def loss_fn(m, x, visible, full):
    p = m(x, visible)
    target = (full - visible).clamp(0, 1)
    return F.binary_cross_entropy_with_logits(p, target) + dice_loss(p, target)


def predict(m, x, v):
    return v + (1 - v) * m(x, v).sigmoid()
