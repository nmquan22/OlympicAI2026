from .common import Encoder, F, backbone, nn, tokens, torch


class SlotAttention(nn.Module):

    def __init__(self, d=128, k=6, steps=3):
        super().__init__()
        self.k = k
        self.steps = steps
        self.mu = nn.Parameter(torch.zeros(1, 1, d))
        self.log_std = nn.Parameter(torch.zeros(1, 1, d))
        self.norm = nn.LayerNorm(d)
        self.sn = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kp = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, x):
        b, n, d = x.shape
        x = self.norm(x)
        k, v = (self.kp(x), self.v(x))
        s = self.mu + self.log_std.exp() * torch.randn(
            b, self.k, d, device=x.device, dtype=x.dtype
        )
        for _ in range(self.steps):
            a = (self.q(self.sn(s)) @ k.transpose(-1, -2) / d**0.5).softmax(1) + 1e-08
            u = a / a.sum(-1, keepdim=True) @ v
            s = self.gru(u.reshape(-1, d), s.reshape(-1, d)).reshape(b, self.k, d)
            s = s + self.mlp(s)
        return (s, a)


class SlotAutoencoder(nn.Module):

    def __init__(self, d=128, k=6, feature_mode=False, teacher=None):
        super().__init__()
        self.enc = Encoder(d)
        self.slot = SlotAttention(d, k)
        self.feature_mode = feature_mode
        c = 128 if feature_mode else 3
        self.teacher = (
            (teacher if teacher is not None else backbone()).requires_grad_(False)
            if feature_mode
            else None
        )
        self.dec = nn.Sequential(
            nn.Linear(d + 2, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, c + 1),
        )

    def forward(self, x):
        f = self.enc(x)
        b, d, h, w = f.shape
        s, _ = self.slot(tokens(f))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        xy = (
            torch.stack([xx, yy], -1)
            .reshape(1, 1, h * w, 2)
            .expand(b, s.shape[1], -1, -1)
        )
        out = self.dec(torch.cat([s[:, :, None].expand(-1, -1, h * w, -1), xy], -1))
        masks = out[..., -1].softmax(1)
        pred = (
            (out[..., :-1] * masks[..., None])
            .sum(1)
            .transpose(1, 2)
            .reshape(b, -1, h, w)
        )
        if self.feature_mode:
            self.teacher.eval()
            with torch.no_grad():
                target = self.teacher(x)["fine"]
        else:
            target = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
        return (pred, target, masks.reshape(b, -1, h, w))


def loss_fn(m, x):
    p, t, _ = m(x)
    return F.mse_loss(p, t)
