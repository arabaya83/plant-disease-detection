"""Classical image segmentation for multi-leaf diagnosis.

This module extracts up to a fixed number of leaf crops from one uploaded
image. It is part of the deployed inference path and deliberately uses simple,
explainable OpenCV heuristics so the segmentation behavior is easy to inspect
and adjust during demos or coursework review.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class LeafSegment:
    """Segmented leaf region produced by the classical CV pipeline.

    Attributes:
        leaf_id: Sequential identifier used in API responses and file names.
        bbox: Bounding box in ``(x0, y0, x1, y1)`` image coordinates.
        crop_bgr: Cropped BGR image that will be passed to the classifier.
    """

    leaf_id: int
    bbox: tuple[int, int, int, int]
    crop_bgr: np.ndarray


class SegmentationService:
    """Extract likely leaf crops using color masking and contour filtering.

    This service is designed for PlantVillage-style imagery and clean demo
    photos. The heuristics favor readability and determinism over robustness to
    cluttered field scenes.
    """

    def __init__(self, max_leaves: int = 5, min_area_ratio: float = 0.01):
        """Initialize segmentation thresholds.

        Args:
            max_leaves: Maximum number of returned leaf regions.
            min_area_ratio: Minimum contour area relative to the full image to
                be considered a valid leaf candidate.
        """
        self.max_leaves = max_leaves
        self.min_area_ratio = min_area_ratio

    def segment_leaves(self, image_bgr: np.ndarray) -> List[LeafSegment]:
        """Segment likely leaf regions from an input image.

        Args:
            image_bgr: Input image in OpenCV BGR format.

        Returns:
            A list of up to ``max_leaves`` segments sorted from largest to
            smallest bounding-box area.
        """
        image_height, image_width = image_bgr.shape[:2]
        image_area = image_height * image_width

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # A wide green mask works well for this prototype because most expected
        # inputs contain leaf tissue against relatively simple backgrounds.
        lower_green = np.array([20, 20, 20], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segments: List[LeafSegment] = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < image_area * self.min_area_ratio:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 10 or box_height <= 10:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            # This permissive circularity filter removes degenerate contours
            # without assuming leaves are close to circular.
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
            if circularity < 0.05:
                continue

            padding = 5
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(image_width, x + box_width + padding)
            y1 = min(image_height, y + box_height + padding)
            leaf_crop = image_bgr[y0:y1, x0:x1].copy()
            segments.append(
                LeafSegment(
                    leaf_id=len(segments) + 1,
                    bbox=(x0, y0, x1, y1),
                    crop_bgr=leaf_crop,
                )
            )

        segments = sorted(segments, key=lambda s: (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]), reverse=True)
        return segments[: self.max_leaves]
