from .common import nn, torch


def entropy(logits):
    p = logits.softmax(-1)
    return -(p * logits.log_softmax(-1)).sum(-1)


def configure_tent(model):
    model.eval()
    model.requires_grad_(False)
    params = []
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.train()
            layer.track_running_stats = False
            layer.running_mean = None
            layer.running_var = None
            layer.requires_grad_(True)
            params.extend(layer.parameters())
    return torch.optim.Adam(params, lr=0.0001)


def adapt(model, opt, x, threshold=None):
    logits = model(x)
    e = entropy(logits)
    loss = (
        e.mean()
        if threshold is None
        else (e * (e.detach() < threshold)).sum()
        / (e.detach() < threshold).sum().clamp_min(1)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    return logits.detach()
