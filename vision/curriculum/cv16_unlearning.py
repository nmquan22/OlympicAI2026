from .common import F, torch


def objective(model, reference, retain_x, retain_y, forget_x, forget_y, alpha=0.1):
    r = model(retain_x)
    f = model(forget_x)
    with torch.no_grad():
        target = reference(retain_x).softmax(-1)
    uniform = -f.log_softmax(-1).mean(-1).mean()
    return (
        F.cross_entropy(r, retain_y)
        + F.kl_div(r.log_softmax(-1), target, reduction="batchmean")
        + alpha * uniform
    )
