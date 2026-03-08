"""Grad-CAM generation service used by inference endpoints."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


class GradCAMService:
    """Produces class-specific saliency overlays for leaf predictions."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register activation/gradient hooks on the configured target layer."""
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_overlay(self, input_tensor: torch.Tensor, image_bgr: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate a color heatmap overlay aligned with the input leaf crop."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
        return overlay

    @staticmethod
    def save_overlay(overlay_bgr: np.ndarray, out_path: Path) -> None:
        """Persist generated Grad-CAM overlay to disk."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay_bgr)
