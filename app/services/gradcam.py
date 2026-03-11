"""Grad-CAM generation used by the deployed inference API.

This module produces explainability overlays for accepted leaf predictions.
It wraps the target-layer hooks and overlay generation logic so the route layer
can request Grad-CAM images without dealing with autograd internals directly.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


class GradCAMService:
    """Generate Grad-CAM overlays for model predictions."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Register hooks required to compute Grad-CAM.

        Args:
            model: Trained classifier used for forward and backward passes.
            target_layer: Convolutional layer whose activations are visualized.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register activation and gradient hooks on the target layer.

        Side Effects:
            Mutates the wrapped model by attaching forward and backward hooks.
        """
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_overlay(self, input_tensor: torch.Tensor, image_bgr: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate a heatmap overlay aligned to a leaf crop.

        Args:
            input_tensor: Preprocessed model input tensor for one leaf crop.
            image_bgr: Original crop image used for visualization.
            class_idx: Optional class index to explain. When omitted, the top
                predicted class is used.

        Returns:
            A BGR image containing the original crop blended with the Grad-CAM
            heatmap.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        channel_weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        class_activation_map = torch.sum(channel_weights * self.activations, dim=1).squeeze(0)
        class_activation_map = torch.relu(class_activation_map)
        class_activation_map = class_activation_map.detach().cpu().numpy()

        if class_activation_map.max() > 0:
            class_activation_map = class_activation_map / class_activation_map.max()

        class_activation_map = cv2.resize(class_activation_map, (image_bgr.shape[1], image_bgr.shape[0]))
        heatmap = np.uint8(255 * class_activation_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
        return overlay

    @staticmethod
    def save_overlay(overlay_bgr: np.ndarray, out_path: Path) -> None:
        """Persist a generated Grad-CAM overlay to disk.

        Args:
            overlay_bgr: Visualization image to save.
            out_path: Destination path for the saved image.

        Side Effects:
            Creates parent directories and writes the image file.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay_bgr)
