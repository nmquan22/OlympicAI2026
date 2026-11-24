from .common import Encoder, F, nn, pool


class World(nn.Module):

    def __init__(self, d=128):
        super().__init__()
        self.enc = Encoder(d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.dec = nn.Sequential(
            nn.Linear(d, d * 4 * 4),
            nn.Unflatten(1, (d, 4, 4)),
            nn.ConvTranspose2d(d, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, frames):
        b, t = frames.shape[:2]
        z = pool(self.enc(frames.flatten(0, 1))).reshape(b, t, -1)
        h, _ = self.rnn(z)
        out = self.dec(h.flatten(0, 1))
        out = F.interpolate(
            out, frames.shape[-2:], mode="bilinear", align_corners=False
        )
        return out.reshape(b, t, 3, *frames.shape[-2:])


def loss_fn(m, video):
    return F.l1_loss(m(video[:, :-1]), video[:, 1:])
