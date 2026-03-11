"""Simple convolutional baseline used for model-comparison experiments.

This architecture is intentionally small and easy to understand. It provides a
from-scratch baseline against which the transfer-learning and hybrid models can
be compared.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Compact convolutional classifier built from scratch.

    The model stacks four convolutional blocks followed by a lightweight MLP
    classifier head. It is not the deployment model; it exists as an
    interpretable baseline for the comparison study.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for an input image batch.

        Args:
            x: Input tensor of shape ``(batch, channels, height, width)``.

        Returns:
            Unnormalized class logits.
        """
        features = self.features(x)
        return self.classifier(features)
