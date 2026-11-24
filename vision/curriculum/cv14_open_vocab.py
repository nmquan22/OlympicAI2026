from .common import Encoder, F, nn, pool, torch, transformer


def load_clip(path):
    from transformers import CLIPModel, CLIPProcessor

    return (
        CLIPModel.from_pretrained(path, local_files_only=True),
        CLIPProcessor.from_pretrained(path, local_files_only=True),
    )


class DualEncoder(nn.Module):

    def __init__(self, vocab=8192, d=128, max_len=64):
        super().__init__()
        self.image = Encoder(d)
        self.text = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.tr = transformer(d)

    def forward(self, x, ids, mask):
        a = F.normalize(pool(self.image(x)), dim=-1)
        z = self.tr(
            self.text(ids) + self.pos[:, : ids.shape[1]],
            src_key_padding_mask=~mask.bool(),
        )
        b = F.normalize(
            (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1),
            dim=-1,
        )
        return a @ b.T / 0.07


def loss_fn(m, x, ids, mask):
    s = m(x, ids, mask)
    y = torch.arange(s.shape[0], device=s.device)
    return (F.cross_entropy(s, y) + F.cross_entropy(s.T, y)) / 2
