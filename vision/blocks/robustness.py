"""Selective perturbations for the supplied classifiers in IOAI Double Agent Dilemma."""
import torch
from torch.nn import functional as F


def selective_perturbation(keep_model, fool_model, images, labels, epsilon=8/255,
                           step_size=1/255, steps=20, keep_weight=2.0):
    """Models include differentiable official preprocessing; images are raw [0,1].

    Returns delta at ORIGINAL image resolution. Put both models in eval mode.
    This is a bounded teaching baseline; the official score also penalizes norm.
    """
    delta = torch.zeros_like(images)
    for _ in range(steps):
        delta.requires_grad_(True)
        changed = (images + delta).clamp(0, 1)
        objective = F.cross_entropy(fool_model(changed), labels) - keep_weight * F.cross_entropy(keep_model(changed), labels)
        gradient, = torch.autograd.grad(objective, delta)
        delta = (delta.detach() + step_size * gradient.sign()).clamp(-epsilon, epsilon)
        delta = (images + delta).clamp(0, 1) - images
    return delta.detach()


@torch.inference_mode()
def selective_success(keep_model, fool_model, images, labels, delta):
    changed = (images + delta).clamp(0, 1)
    return keep_model(changed).argmax(1).eq(labels) & fool_model(changed).argmax(1).ne(labels)
