"""Model loading and leaf-level prediction for deployed inference.

This module owns the runtime classifier used by the FastAPI application. It
loads class metadata and model weights, applies the same normalization used
during training, and converts raw logits into the user-facing prediction fields
required by the frontend.
"""

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
    """Structured prediction returned for one segmented leaf.

    Attributes:
        class_label: Raw dataset class label produced by the model.
        confidence: Softmax confidence for the winning class.
        crop_name: Human-readable crop name derived from the class label.
        disease_name: Human-readable disease name for the UI.
        healthy_or_diseased: High-level health label used for badges.
        short_description: Short explanatory text shown in the UI.
        class_index: Integer class index used for Grad-CAM targeting.
    """

    class_label: str
    confidence: float
    crop_name: str
    disease_name: str
    healthy_or_diseased: str
    short_description: str
    class_index: int


class InferenceService:
    """Load a trained classifier and perform leaf-level predictions.

    The service is instantiated once at application startup and reused across
    requests. This avoids reloading model weights for every inference call.
    """

    def __init__(
        self,
        model_name: str,
        weights_path: str,
        class_names_path: str,
        image_size: int,
        device: str | None = None,
        strict_loading: bool = True,
    ):
        """Initialize the runtime inference service.

        Args:
            model_name: Architecture identifier: ``cnn``, ``mobilenet``, or
                ``hybrid``.
            weights_path: Filesystem path to the trained checkpoint.
            class_names_path: JSON file mapping class indices to labels.
            image_size: Target square image size used for preprocessing.
            device: Optional explicit torch device override.
            strict_loading: Whether missing weights/class metadata should raise
                an exception instead of falling back to placeholder defaults.

        Raises:
            FileNotFoundError: If required weights or class metadata are missing
                and ``strict_loading`` is enabled.
        """
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
        """Load class labels from disk.

        Args:
            class_names_path: Path to the JSON file storing class-index mapping.

        Returns:
            Ordered list of class labels aligned with model output indices.

        Raises:
            FileNotFoundError: If the file is missing and strict loading is
                enabled.
        """
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

        # The fallback list keeps the module inspectable in contexts where full
        # artifacts are absent, but strict mode remains the default for the app.
        return [
            "Apple___healthy",
            "Apple___Apple_scab",
            "Tomato___healthy",
            "Tomato___Late_blight",
            "Potato___healthy",
            "Potato___Late_blight",
        ]

    def _build_model(self, model_name: str, num_classes: int) -> torch.nn.Module:
        """Instantiate the requested classifier architecture.

        Args:
            model_name: Requested architecture name.
            num_classes: Number of output classes required by the checkpoint.

        Returns:
            Torch module moved onto the configured device.
        """
        normalized_model_name = model_name.lower()
        if normalized_model_name == "cnn":
            model = SimpleCNN(num_classes=num_classes)
        elif normalized_model_name == "mobilenet":
            model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
        else:
            model = HybridPlantDiseaseModel(num_classes=num_classes, pretrained_backbone=False)
        return model.to(self.device)

    def _load_weights(self, weights_path: str) -> bool:
        """Load model weights from a checkpoint file.

        Args:
            weights_path: Path to the checkpoint to load.

        Returns:
            ``True`` when weights were loaded successfully, otherwise ``False``
            if the checkpoint file is missing.
        """
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
        """Predict a label for one segmented leaf crop.

        Args:
            leaf_bgr: Leaf crop in BGR channel order.

        Returns:
            A tuple of the structured :class:`Prediction` and the normalized
            input tensor used to generate it. The tensor is returned so the API
            can reuse it for Grad-CAM without preprocessing twice.
        """
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
        """Return the layer whose activations should drive Grad-CAM.

        Returns:
            The convolutional feature layer that best represents the model's
            final spatial features for Grad-CAM visualization.
        """
        # Use the last convolutional feature block so the overlay reflects the
        # model's final spatial reasoning before classification.
        if hasattr(self.model, "mobilenet"):
            return self.model.mobilenet.features[-1]
        if hasattr(self.model, "features"):
            return self.model.features[-1]
        return list(self.model.modules())[-1]
