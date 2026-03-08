"""Classical OpenCV leaf segmentation for multi-leaf images."""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class LeafSegment:
    """Represents one segmented leaf crop and its original-image bbox."""

    leaf_id: int
    bbox: tuple[int, int, int, int]
    crop_bgr: np.ndarray


class SegmentationService:
    """Extracts up to N valid leaf regions using color + contour filtering."""

    def __init__(self, max_leaves: int = 5, min_area_ratio: float = 0.01):
        self.max_leaves = max_leaves
        self.min_area_ratio = min_area_ratio

    def segment_leaves(self, image_bgr: np.ndarray) -> List[LeafSegment]:
        """Return sorted leaf segments from largest to smallest area."""
        h, w = image_bgr.shape[:2]
        img_area = h * w

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 20, 20], dtype=np.uint8)
        upper_green = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segments: List[LeafSegment] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * self.min_area_ratio:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 10 or bh <= 10:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < 0.05:
                continue

            pad = 5
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad)
            y1 = min(h, y + bh + pad)
            crop = image_bgr[y0:y1, x0:x1].copy()
            segments.append(LeafSegment(leaf_id=len(segments) + 1, bbox=(x0, y0, x1, y1), crop_bgr=crop))

        segments = sorted(segments, key=lambda s: (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]), reverse=True)
        return segments[: self.max_leaves]
