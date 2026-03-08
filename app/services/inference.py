"""Model loading and leaf-level inference service."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torchvision import transforms

from app.services.disease_info import get_description, parse_label
from ml.src.models.cnn_baseline import SimpleCNN
from ml.src.models.hybrid_model import HybridPlantDiseaseModel
from ml.src.models.mobilenet_baseline import build_mobilenet_v2


@dataclass
class Prediction:
    """Structured prediction payload used by API response formatter."""

    class_label: str
    confidence: float
    crop_name: str
    disease_name: str
    healthy_or_diseased: str
    short_description: str
    class_index: int


class InferenceService:
    """Loads selected model/weights and runs normalized leaf classification."""

    def __init__(
        self,
        model_name: str,
        weights_path: str,
        class_names_path: str,
        image_size: int,
        device: str | None = None,
        strict_loading: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.strict_loading = strict_loading
        self.class_names = self._load_class_names(class_names_path)
        self.model = self._build_model(model_name=model_name, num_classes=len(self.class_names))
        self.model_loaded = self._load_weights(weights_path)
        if self.strict_loading and not self.model_loaded:
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_class_names(self, class_names_path: str) -> List[str]:
        """Load class names from JSON; optionally fall back to a minimal list."""
        path = Path(class_names_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                return [v for _, v in sorted(payload.items(), key=lambda x: int(x[0]))]

        if self.strict_loading:
            raise FileNotFoundError(f"Class names file not found at: {class_names_path}")

        return [
            "Apple___healthy",
            "Apple___Apple_scab",
            "Tomato___healthy",
            "Tomato___Late_blight",
            "Potato___healthy",
            "Potato___Late_blight",
        ]

    def _build_model(self, model_name: str, num_classes: int) -> torch.nn.Module:
        """Factory for supported classifier architectures."""
        model_name = model_name.lower()
        if model_name == "cnn":
            model = SimpleCNN(num_classes=num_classes)
        elif model_name == "mobilenet":
            model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
        else:
            model = HybridPlantDiseaseModel(num_classes=num_classes, pretrained_backbone=False)
        return model.to(self.device)

    def _load_weights(self, weights_path: str) -> bool:
        """Load model checkpoint state and report whether load succeeded."""
        path = Path(weights_path)
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        return True

    @torch.no_grad()
    def predict(self, leaf_bgr: np.ndarray) -> tuple[Prediction, torch.Tensor]:
        """Predict class for one leaf crop and return model input tensor."""
        leaf_rgb = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(leaf_rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

        class_idx = int(idx.item())
        confidence = float(conf.item())
        label = self.class_names[class_idx] if class_idx < len(self.class_names) else "Unknown___Unknown"
        crop, disease, hod = parse_label(label)

        pred = Prediction(
            class_label=label,
            confidence=confidence,
            crop_name=crop,
            disease_name=disease,
            healthy_or_diseased=hod,
            short_description=get_description(label, hod),
            class_index=class_idx,
        )
        return pred, tensor

    def get_gradcam_target_layer(self):
        """Return the conv layer used by Grad-CAM for this model."""
        # MobileNet branch conv head in hybrid model; fallback for other models.
        if hasattr(self.model, "mobilenet"):
            return self.model.mobilenet.features[-1]
        if hasattr(self.model, "features"):
            return self.model.features[-1]
        return list(self.model.modules())[-1]
