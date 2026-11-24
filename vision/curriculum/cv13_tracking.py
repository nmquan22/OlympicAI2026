from .common import Encoder, F, nn, torch


class Tracker(nn.Module):

    def __init__(self, d=128):
        super().__init__()
        self.enc = Encoder(d)

    def forward(self, video, points):
        b, t, c, h, w = video.shape
        f = self.enc(video.flatten(0, 1))
        _, d, hh, ww = f.shape
        f = f.reshape(b, t, d, hh, ww)
        q = (
            F.grid_sample(f[:, 0], points[:, :, None], align_corners=False)
            .squeeze(-1)
            .transpose(1, 2)
        )
        z = F.normalize(f.flatten(3), dim=2)
        corr = torch.einsum("bnd,btdm->btnm", F.normalize(q, dim=-1), z) / 0.07
        yy, xx = torch.meshgrid(
            (torch.arange(hh, device=f.device) + 0.5) * 2 / hh - 1,
            (torch.arange(ww, device=f.device) + 0.5) * 2 / ww - 1,
            indexing="ij",
        )
        xy = torch.stack([xx, yy], -1).reshape(-1, 2)
        return corr.softmax(-1) @ xy


def loss_fn(m, v, p, y, valid):
    return ((m(v, p) - y).abs().sum(-1) * valid).sum() / valid.sum().clamp_min(1)
