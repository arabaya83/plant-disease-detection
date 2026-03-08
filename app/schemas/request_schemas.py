"""Pydantic schemas for inference request options."""

from pydantic import BaseModel, Field


class InferOptions(BaseModel):
    """Optional runtime overrides for threshold and leaf count limits."""

    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_leaves: int = Field(default=5, ge=1, le=5)
