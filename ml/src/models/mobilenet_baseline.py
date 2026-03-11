"""MobileNetV2 baseline used for comparison and deployment.

This model is the project's primary transfer-learning baseline and the selected
deployment architecture. The helper function here standardizes how the
classifier head is replaced for the PlantVillage class count.
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate MobileNetV2 with a task-specific classifier head.

    Args:
        num_classes: Number of output classes for the final linear layer.
        pretrained: Whether to initialize the backbone with ImageNet weights.

    Returns:
        Configured MobileNetV2 model ready for training or checkpoint loading.
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
