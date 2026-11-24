from .common import Encoder, F, nn, pool, torch


class Forensics(nn.Module):

    def __init__(self, d=128, mode="fusion"):
        super().__init__()
        self.enc = Encoder(d)
        self.freq = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1)
        )
        self.mode = mode
        self.head = nn.Linear(d + 32 if mode == "fusion" else d, 2)

    def forward(self, x):
        f = pool(self.enc(x))
        spec = torch.fft.fftshift(torch.fft.fft2(x.float()).abs(), dim=(-2, -1)).log1p()
        return self.head(
            torch.cat([f, pool(self.freq(spec))], 1) if self.mode == "fusion" else f
        )


def loss_fn(m, x, y):
    return F.cross_entropy(m(x), y)
