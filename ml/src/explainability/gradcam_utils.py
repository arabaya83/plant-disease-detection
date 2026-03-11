"""Standalone Grad-CAM utilities for offline experiments and notebooks.

This module mirrors the core explainability logic used by the deployed API but
keeps it available in a simpler form for exploratory scripts, report notebooks,
and one-off visual analysis.
"""

import cv2
import numpy as np
import torch


class GradCAM:
    """Generate class activation maps for a configured target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize the Grad-CAM helper and register hooks.

        Args:
            model: Trained model used for forward and backward passes.
            target_layer: Layer whose activations should drive the heatmap.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks required for Grad-CAM."""
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        """Cache target-layer activations from the forward pass."""
        del module, inputs
        self.activations = output.detach()

    def _backward_hook(self, module, grad_inputs, grad_outputs) -> None:
        """Cache target-layer gradients from the backward pass."""
        del module, grad_inputs
        self.gradients = grad_outputs[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Compute a normalized class activation map.

        Args:
            input_tensor: Preprocessed input tensor for one image.
            class_idx: Optional class index to explain. When omitted, the top
                predicted class is used.

        Returns:
            Normalized 2D class activation map as a NumPy array.
        """
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        channel_weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        class_activation_map = torch.sum(channel_weights * self.activations, dim=1).squeeze(0)
        class_activation_map = torch.relu(class_activation_map)
        class_activation_map = class_activation_map.cpu().numpy()
        if class_activation_map.max() > 0:
            class_activation_map /= class_activation_map.max()
        return class_activation_map


def overlay_heatmap(cam: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """Blend a normalized class activation map into the original image.

    Args:
        cam: Normalized class activation map from :meth:`GradCAM.generate`.
        image_bgr: Original BGR image used as the background.

    Returns:
        BGR image containing the original image blended with the heatmap.
    """
    resized_cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(resized_cam * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
