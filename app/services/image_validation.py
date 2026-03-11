"""Frame-level validation for uploaded plant images.

This module implements the first gate in the online inference pipeline. It
uses a lightweight HSV green-ratio heuristic to reject obviously invalid
uploads before the system spends time on segmentation, model inference, and
Grad-CAM generation.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ValidationResult:
    """Validation outcome for one uploaded image.

    Attributes:
        is_valid: Whether the image contains enough leaf-like content to
            continue into segmentation.
        message: User-facing explanation of the validation decision.
    """

    is_valid: bool
    message: str


class LeafValidationService:
    """Detect whether an uploaded image likely contains a plant leaf.

    The service intentionally uses a simple heuristic instead of a learned
    detector so invalid photos can be rejected cheaply and deterministically.
    """

    def __init__(self, min_green_ratio: float = 0.02):
        """Initialize the validator.

        Args:
            min_green_ratio: Minimum fraction of green-like pixels required for
                an image to be treated as a plausible leaf photo.
        """
        self.min_green_ratio = min_green_ratio

    def validate(self, image_bgr: np.ndarray) -> ValidationResult:
        """Evaluate whether an image contains enough leaf-like content.

        Args:
            image_bgr: Input image loaded in OpenCV BGR channel order.

        Returns:
            A :class:`ValidationResult` describing whether the image should
            continue through the inference pipeline.
        """
        if image_bgr is None or image_bgr.size == 0:
            return ValidationResult(False, "Invalid image data.")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # The HSV range is intentionally broad so healthy and diseased leaves
        # still register as "leaf-like" even when color shifts are present.
        lower_green = np.array([25, 25, 25], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = float(np.count_nonzero(mask)) / float(mask.size)

        if green_ratio < self.min_green_ratio:
            return ValidationResult(False, "No leaf detected. Please retake the photo.")
        return ValidationResult(True, "Leaf-like content detected.")
