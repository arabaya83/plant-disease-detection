"""MobileNetV2 transfer-learning baseline."""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate MobileNetV2 and replace classifier head for target classes."""
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
