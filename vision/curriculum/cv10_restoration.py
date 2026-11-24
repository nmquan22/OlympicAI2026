from .common import F, nn, pool


class Restoration(nn.Module):

    def __init__(self, d=64, condition=True):
        super().__init__()
        self.condition = condition
        self.enc = nn.Conv2d(3, d, 3, padding=1)
        self.prompt = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.body = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(d, 3, 3, padding=1)

    def forward(self, x):
        f = self.enc(x)
        if self.condition:
            scale, bias = self.prompt(pool(f)).chunk(2, -1)
            f = f * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return x + self.out(self.body(f) + f)


def loss_fn(m, x, y):
    return F.l1_loss(m(x), y)
