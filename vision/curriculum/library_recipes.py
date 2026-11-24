"""Optional library APIs. All loaders are offline by default."""

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor


def load_local(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def resnet_pyramid(path):
    model = load_local(models.resnet18(weights=None), path)
    return create_feature_extractor(
        model, {"layer1": "p2", "layer2": "p3", "layer3": "p4", "layer4": "p5"}
    )


def swin_pyramid(path):
    model = load_local(models.swin_t(weights=None), path)
    return create_feature_extractor(
        model,
        {
            "features.1": "p2",
            "features.3": "p3",
            "features.5": "p4",
            "features.7": "p5",
        },
    )


def vit_tokens(path):
    from transformers import AutoModel

    return AutoModel.from_pretrained(path, local_files_only=True)


def small_vit():
    from transformers import ViTConfig, ViTModel

    return ViTModel(
        ViTConfig(
            image_size=224,
            patch_size=16,
            hidden_size=192,
            num_hidden_layers=6,
            num_attention_heads=3,
            intermediate_size=768,
        )
    )


def small_token_lm(vocab=256):
    from transformers import GPT2Config, GPT2LMHeadModel

    return GPT2LMHeadModel(
        GPT2Config(vocab_size=vocab, n_embd=192, n_head=3, n_layer=4, n_positions=1024)
    )


def segmentation(classes=2):
    return models.segmentation.deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=classes
    )


def detector(classes=2):
    return models.detection.maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=classes
    )


def optical_flow():
    return models.optical_flow.raft_small(weights=None)


def augment():
    from torchvision.transforms import v2

    return v2.Compose(
        [v2.ToImage(), v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True)]
    )
