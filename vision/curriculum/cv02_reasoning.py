from .common import Encoder, F, nn, pool, torch, transformer


class Reasoner(nn.Module):

    def __init__(self, d=128, mode="relation"):
        super().__init__()
        self.enc = Encoder(d)
        self.mode = mode
        self.pos = nn.Parameter(torch.randn(1, 9, d) * 0.02)
        self.rel = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.tr = transformer(d)
        self.head = nn.Linear(d, 1)
        self.base = nn.Sequential(
            nn.Linear(9 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, context, choices):
        b, a, c, h, w = choices.shape
        z = pool(self.enc(torch.cat([context, choices], 1).flatten(0, 1))).reshape(
            b, 8 + a, -1
        )
        v = (
            torch.cat(
                [z[:, :8, None].expand(-1, -1, a, -1).transpose(1, 2), z[:, 8:, None]],
                2,
            ).flatten(0, 1)
            + self.pos
        )
        if self.mode == "baseline":
            u = self.base(v.flatten(1))
        elif self.mode == "relation":
            u = self.rel(
                torch.cat(
                    [
                        v[:, :, None].expand(-1, -1, 9, -1),
                        v[:, None].expand(-1, 9, -1, -1),
                    ],
                    -1,
                )
            ).mean((1, 2))
        else:
            u = self.tr(v)[:, -1]
        return self.head(u).reshape(b, a)


def loss_fn(m, c, a, y):
    return F.cross_entropy(m(c, a), y)
