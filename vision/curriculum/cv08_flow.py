from .common import F, nn, torch


class Velocity(nn.Module):

    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, d, 3, padding=1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d, 3, 3, padding=1),
        )

    def forward(self, x, t):
        return self.net(
            torch.cat([x, t[:, None, None, None].expand(-1, 1, *x.shape[-2:])], 1)
        )


def loss_fn(m, x):
    z = torch.randn_like(x)
    t = torch.rand(x.shape[0], device=x.device)
    v = t[:, None, None, None]
    return F.mse_loss(m((1 - v) * z + v * x, t), x - z)


@torch.no_grad()
def sample(m, z, steps=32):
    for i in range(steps):
        z = z + m(z, torch.full((z.shape[0],), i / steps, device=z.device)) / steps
    return z
