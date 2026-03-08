"""Leaf-content validation service using simple color heuristics."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ValidationResult:
    """Validation output for incoming frame-level image checks."""

    is_valid: bool
    message: str


class LeafValidationService:
    """Detects whether an image likely contains sufficient leaf content."""

    def __init__(self, min_green_ratio: float = 0.02):
        self.min_green_ratio = min_green_ratio

    def validate(self, image_bgr: np.ndarray) -> ValidationResult:
        """Run green-ratio heuristic in HSV space and return validation result."""
        if image_bgr is None or image_bgr.size == 0:
            return ValidationResult(False, "Invalid image data.")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 25, 25], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = float(np.count_nonzero(mask)) / float(mask.size)

        if green_ratio < self.min_green_ratio:
            return ValidationResult(False, "No leaf detected. Please retake the photo.")
        return ValidationResult(True, "Leaf-like content detected.")
