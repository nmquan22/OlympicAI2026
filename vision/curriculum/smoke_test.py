import torch
from torch import nn
from torchvision.models import resnet18
from . import cv01_slots as a
from . import cv02_reasoning as b
from . import cv03_visual_context as c
from . import cv04_composition as d
from . import cv05_subtle as e
from . import cv06_adapt as f
from . import cv07_tokenization as g
from . import cv08_flow as h
from . import cv09_forensics as i
from . import cv10_restoration as j
from . import cv11_matching as k
from . import cv12_amodal as l
from . import cv13_tracking as m
from . import cv14_open_vocab as n
from . import cv15_world as o
from . import cv16_unlearning as p

torch.set_num_threads(2)
torch.manual_seed(42)
x = torch.rand(2, 3, 32, 32)
y = torch.tensor([0, 1])
mask = (torch.rand(2, 1, 32, 32) > 0.5).float()
video = torch.rand(2, 4, 3, 32, 32)


def check(name, model, fn, *args):
    loss = fn(model, *args)
    loss.backward()
    print(name, round(loss.item(), 4), flush=True)


for mode in [False, True]:
    check("01 slots " + str(mode), a.SlotAutoencoder(feature_mode=mode), a.loss_fn, x)
for mode in ["baseline", "relation", "transformer"]:
    check(
        "02 " + mode,
        b.Reasoner(mode=mode),
        b.loss_fn,
        torch.rand(2, 8, 3, 32, 32),
        torch.rand(2, 4, 3, 32, 32),
        y,
    )
for mode in ["prototype", "cross"]:
    check(
        "03 " + mode,
        c.VisualContext(mode=mode),
        c.loss_fn,
        x[:, None],
        mask[:, None],
        x,
        mask,
    )
for mode in ["factorized", "joint"]:
    check(
        "04 " + mode,
        d.Composition(mode=mode),
        d.loss_fn,
        x,
        torch.tensor([[0, 0], [1, 1], [0, 1]]),
        y,
    )
check("05 pulse", e.PulseNet(), e.loss_fn, video, torch.rand(2, 4))
print("05 chrom", e.heart_rate(e.chrom(torch.rand(2, 120, 3, 8, 8))))
model = resnet18(weights=None, num_classes=2)
opt = f.configure_tent(model)
print("06 adapt", f.adapt(model, opt, x).shape)
check("07 vqvae", g.VQVAE(), g.loss_fn, x)
check("08 flow", h.Velocity(), h.loss_fn, x)
for mode in ["rgb", "fusion"]:
    check("09 " + mode, i.Forensics(mode=mode), i.loss_fn, x, y)
for mode in [False, True]:
    check("10 " + str(mode), j.Restoration(condition=mode), j.loss_fn, x, x)
for mode in [False, True]:
    check(
        "11 " + str(mode),
        k.Matcher(context=mode),
        k.loss_fn,
        x,
        x,
        torch.arange(16).repeat(2, 1),
    )
check("12 amodal", l.Amodal(), l.loss_fn, x, mask, torch.ones_like(mask))
check(
    "13 tracking",
    m.Tracker(),
    m.loss_fn,
    video,
    torch.rand(2, 3, 2) * 2 - 1,
    torch.rand(2, 4, 3, 2) * 2 - 1,
    torch.ones(2, 4, 3),
)
check(
    "14 dual encoder",
    n.DualEncoder(),
    n.loss_fn,
    x,
    torch.randint(0, 8192, (2, 8)),
    torch.ones(2, 8, dtype=torch.bool),
)
check("15 world", o.World(), o.loss_fn, video)
model = nn.Sequential(nn.Flatten(), nn.Linear(3072, 2))
ref = nn.Sequential(nn.Flatten(), nn.Linear(3072, 2))
loss = p.objective(model, ref, x, y, x, y)
loss.backward()
print("16 unlearn", loss.item())
