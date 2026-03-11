"""Hybrid classifier that fuses transfer-learned and custom CNN features.

This model combines a MobileNetV2 backbone with a lightweight residual branch.
It exists to test whether multi-branch feature fusion improves over the
transfer-learning baseline enough to justify the added complexity.
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ResidualBlock(nn.Module):
    """Residual refinement block used inside the custom branch."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual transform and return the fused activation."""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class HybridPlantDiseaseModel(nn.Module):
    """Feature-fusion model with MobileNet and residual branches."""

    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        """Initialize the hybrid classifier.

        Args:
            num_classes: Number of output classes for classification.
            pretrained_backbone: Whether to initialize MobileNet with ImageNet
                weights.
        """
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.mobilenet = mobilenet_v2(weights=weights)
        self.mobilenet_backbone = self.mobilenet.features
        self.mobilenet_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.res_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 + 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract both branch features, fuse them, and produce logits.

        Args:
            x: Input image batch.

        Returns:
            Unnormalized class logits for each image.
        """
        mobile_features = self.mobilenet_backbone(x)
        mobile_features = self.mobilenet_pool(mobile_features)
        residual_features = self.res_branch(x)

        fused_features = torch.cat([mobile_features, residual_features], dim=1)
        return self.head(fused_features)
