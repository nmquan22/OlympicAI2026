from .common import F, nn, torch


def rgb_trace(video):
    return video.mean((-2, -1))


def chrom(video):
    z = rgb_trace(video)
    z = z / z.mean(1, keepdim=True).clamp_min(1e-06) - 1
    r, g, b = z.unbind(-1)
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    return x - x.std(1, keepdim=True) / y.std(1, keepdim=True).clamp_min(1e-06) * y


def heart_rate(signal, fps=30):
    s = signal - signal.mean(-1, keepdim=True)
    p = torch.fft.rfft(s * torch.hann_window(s.shape[-1], device=s.device)).abs()
    f = torch.fft.rfftfreq(s.shape[-1], 1 / fps, device=s.device)
    return 60 * f[p.masked_fill((f < 0.7) | (f > 3.5), -1).argmax(-1)]


class PulseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.AvgPool3d((1, 2, 2)),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 1, 1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).mean((-2, -1)).squeeze(1)


def loss_fn(m, x, y):
    p = m(x)
    p = p - p.mean(-1, keepdim=True)
    y = y - y.mean(-1, keepdim=True)
    return (1 - F.cosine_similarity(p, y)).mean() + 0.1 * F.mse_loss(
        F.normalize(torch.fft.rfft(p).abs()), F.normalize(torch.fft.rfft(y).abs())
    )
