from .common import F, nn, tokens


class VQVAE(nn.Module):

    def __init__(self, d=64, k=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1), nn.GELU(), nn.Conv2d(d, d, 4, 2, 1)
        )
        self.code = nn.Embedding(k, d)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, d, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(d, 3, 4, 2, 1),
        )

    def forward(self, x):
        z = self.enc(x)
        t = tokens(z)
        c = self.code.weight
        dist = t.square().sum(-1, keepdim=True) + c.square().sum(-1) - 2 * t @ c.T
        idx = dist.argmin(-1)
        q = self.code(idx).transpose(1, 2).reshape_as(z)
        vq = F.mse_loss(q, z.detach()) + 0.25 * F.mse_loss(z, q.detach())
        return (self.dec(z + (q - z).detach()), idx, vq)


def loss_fn(m, x):
    p, _, vq = m(x)
    return F.l1_loss(p, x) + vq
