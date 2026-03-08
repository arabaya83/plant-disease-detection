"""Pydantic response schemas used by inference endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class LeafResult(BaseModel):
    """Leaf-level diagnosis payload returned to frontend/API consumers."""

    leaf_id: int
    crop_name: str
    disease_name: str
    confidence: float
    healthy_or_diseased: str
    short_description: str
    heatmap_path: Optional[str] = None


class InferResponse(BaseModel):
    """Top-level response schema for `/infer`."""

    status: str
    message: str
    image_id: Optional[str] = None
    latency_ms: Optional[float] = None
    total_leaves_detected: int = 0
    results: List[LeafResult] = Field(default_factory=list)
