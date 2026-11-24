from .common import Cross, Encoder, F, dice_loss, nn, tokens


class VisualContext(nn.Module):

    def __init__(self, d=128, mode="prototype"):
        super().__init__()
        self.enc = Encoder(d)
        self.mode = mode
        self.mask_emb = nn.Linear(1, d)
        self.cross = Cross(d)
        self.head = nn.Linear(d, 1)

    def forward(self, s, y, q):
        b, k = s.shape[:2]
        sf = self.enc(s.flatten(0, 1))
        qf = self.enc(q)
        h, w = qf.shape[-2:]
        st = tokens(sf).reshape(b, -1, sf.shape[1])
        qt = tokens(qf)
        mask = F.interpolate(
            y.flatten(0, 1).float(), sf.shape[-2:], mode="nearest"
        ).reshape(b, -1, 1)
        if self.mode == "prototype":
            fg = (st * mask).sum(1) / mask.sum(1).clamp_min(1)
            bg = (st * (1 - mask)).sum(1) / (1 - mask).sum(1).clamp_min(1)
            score = 10 * (
                F.normalize(qt, dim=-1)
                * (F.normalize(fg, dim=-1) - F.normalize(bg, dim=-1))[:, None]
            ).sum(-1)
        else:
            score = self.head(self.cross(qt, st + self.mask_emb(mask))).squeeze(-1)
        return F.interpolate(
            score.reshape(b, 1, h, w),
            q.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def loss_fn(m, s, y, q, t):
    p = m(s, y, q)
    return F.binary_cross_entropy_with_logits(p, t.float()) + dice_loss(p, t)
