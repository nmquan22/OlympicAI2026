from .common import Encoder, F, nn, pool, torch


class Composition(nn.Module):

    def __init__(self, n_attr=10, n_obj=12, d=128, mode="factorized"):
        super().__init__()
        self.enc = Encoder(d)
        self.a = nn.Linear(d, n_attr)
        self.o = nn.Linear(d, n_obj)
        self.aemb = nn.Embedding(n_attr, d)
        self.oemb = nn.Embedding(n_obj, d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.mode = mode

    def forward(self, x, pairs):
        z = pool(self.enc(x))
        a, o = (self.a(z), self.o(z))
        score = a[:, pairs[:, 0]] + o[:, pairs[:, 1]]
        if self.mode == "joint":
            p = self.fuse(
                torch.cat([self.aemb(pairs[:, 0]), self.oemb(pairs[:, 1])], -1)
            )
            score = score + 10 * F.normalize(z, dim=-1) @ F.normalize(p, dim=-1).T
        return (score, a, o)


def loss_fn(m, x, pairs, idx):
    s, a, o = m(x, pairs)
    y = pairs[idx]
    return (
        F.cross_entropy(s, idx)
        + F.cross_entropy(a, y[:, 0])
        + F.cross_entropy(o, y[:, 1])
    )
