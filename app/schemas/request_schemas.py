"""Optional request-side schemas for future inference customization.

The current browser client posts images directly as multipart uploads, so this
module is intentionally small. It exists to document the kinds of runtime
options the API may safely accept in the future without embedding validation
rules directly inside route handlers.
"""

from pydantic import BaseModel, Field


class InferOptions(BaseModel):
    """Validated inference options for advanced or future clients.

    Attributes:
        confidence_threshold: Minimum model confidence required before a
            prediction is returned to the user.
        max_leaves: Maximum number of segmented leaves to process from a
            single image.
    """

    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    max_leaves: int = Field(default=5, ge=1, le=5)
