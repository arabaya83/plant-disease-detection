"""Response models returned by the inference and analytics API.

These schemas make the deployed API contract explicit for the browser client,
reviewers, and future maintainers. Keeping response structure in one module
reduces the risk of route handlers drifting away from the documented payloads.
"""

from typing import Optional

from pydantic import BaseModel, Field


class LeafResult(BaseModel):
    """Diagnosis details for one segmented leaf.

    Attributes:
        leaf_id: Sequential identifier assigned during segmentation.
        crop_name: Human-readable crop name parsed from the class label.
        disease_name: Human-readable disease name or "No disease detected".
        confidence: Softmax confidence for the predicted class.
        healthy_or_diseased: High-level health summary used by the frontend.
        short_description: Short educational message shown to the user.
        heatmap_path: Optional URL path to the saved Grad-CAM overlay image.
    """

    leaf_id: int
    crop_name: str
    disease_name: str
    confidence: float
    healthy_or_diseased: str
    short_description: str
    heatmap_path: Optional[str] = None


class InferResponse(BaseModel):
    """Top-level payload returned by the `/infer` endpoint.

    Attributes:
        status: Pipeline outcome, such as ``ok``, ``invalid``, or
            ``low_confidence``.
        message: User-facing summary of the inference result.
        image_id: Identifier for the stored upload, when available.
        latency_ms: End-to-end server-side processing latency in milliseconds.
        total_leaves_detected: Number of valid segments produced before
            confidence filtering.
        results: Accepted per-leaf predictions and Grad-CAM metadata.
    """

    status: str
    message: str
    image_id: Optional[str] = None
    latency_ms: Optional[float] = None
    total_leaves_detected: int = 0
    results: list[LeafResult] = Field(default_factory=list)
