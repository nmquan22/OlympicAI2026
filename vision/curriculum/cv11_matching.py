from .common import Cross, Encoder, F, nn, tokens, torch


class Matcher(nn.Module):

    def __init__(self, d=128, context=True):
        super().__init__()
        self.enc = Encoder(d)
        self.cross = Cross(d)
        self.context = context

    def forward(self, a, b):
        a, b = (tokens(self.enc(a)), tokens(self.enc(b)))
        if self.context:
            a, b = (self.cross(a, b), self.cross(b, a))
        return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2) / 0.07


def loss_fn(m, a, b, target):
    return F.cross_entropy(m(a, b).transpose(1, 2), target, ignore_index=-100)


def soft_coordinates(corr, h, w):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=corr.device),
        torch.arange(w, device=corr.device),
        indexing="ij",
    )
    xy = torch.stack([xx, yy], -1).reshape(-1, 2).to(corr.dtype)
    return corr.softmax(-1) @ xy
