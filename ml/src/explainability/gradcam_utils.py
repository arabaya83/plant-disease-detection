"""Standalone Grad-CAM utilities for experiments and notebooks."""

import cv2
import numpy as np
import torch


class GradCAM:
    """Generate class activation maps for a configured target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks needed for Grad-CAM."""
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        """Cache activations from target layer forward pass."""
        del module, inp
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        """Cache gradients from target layer backward pass."""
        del module, grad_in
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None):
        """Compute normalized CAM for selected class index."""
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(cam: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """Blend normalized CAM into original BGR image."""
    cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
