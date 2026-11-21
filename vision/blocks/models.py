import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign


def conv_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.shallow = conv_block(3, 16)
        self.deep = conv_block(16, 32)
        self.decode = conv_block(48, 16)
        self.head = nn.Conv2d(16, classes, 1)

    def forward(self, images):
        skip = self.shallow(images)
        deep = self.deep(F.max_pool2d(skip, 2))
        high = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decode(torch.cat((skip, high), dim=1)))


def tiny_detector(classes, size=160):
    """One feature level, three anchor scales. classes includes background."""
    backbone = nn.Sequential(conv_block(3, 16), nn.MaxPool2d(2),
                             conv_block(16, 32), nn.MaxPool2d(2), conv_block(32, 32), nn.MaxPool2d(2))
    backbone.out_channels = 32
    anchors = AnchorGenerator(sizes=((12, 32, 72),), aspect_ratios=((0.5, 1.0, 2.0),))
    pool = MultiScaleRoIAlign(["0"], output_size=3, sampling_ratio=2)
    return FasterRCNN(backbone, num_classes=None, min_size=size, max_size=size,
                      rpn_anchor_generator=anchors, box_roi_pool=pool,
                      box_head=TwoMLPHead(32 * 3 * 3, 128), box_predictor=FastRCNNPredictor(128, classes),
                      rpn_pre_nms_top_n_train=300, rpn_post_nms_top_n_train=120,
                      rpn_pre_nms_top_n_test=180, rpn_post_nms_top_n_test=80)
