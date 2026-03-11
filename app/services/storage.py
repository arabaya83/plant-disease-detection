"""Filesystem helpers for uploaded inputs and generated artifacts.

The deployed API stores original uploads and Grad-CAM overlays under
``app/static`` so they can be served back to the browser. This module
centralizes naming and directory creation to keep route handlers focused on
pipeline orchestration instead of path management.
"""

import uuid
from pathlib import Path

from fastapi import UploadFile


class StorageService:
    """Manage runtime file paths for uploads and derived images."""

    def __init__(self, upload_dir: str, output_dir: str):
        """Create storage directories used by the online inference flow.

        Args:
            upload_dir: Directory where original uploaded images are stored.
            output_dir: Directory where Grad-CAM overlay images are written.
        """
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload_file: UploadFile) -> tuple[str, Path]:
        """Persist an uploaded image and assign it a stable identifier.

        Args:
            upload_file: Multipart upload provided by FastAPI.

        Returns:
            A tuple of ``(image_id, path)`` for the saved upload.

        Side Effects:
            Writes the uploaded file contents to disk.
        """
        image_id = uuid.uuid4().hex
        suffix = Path(upload_file.filename or "upload.jpg").suffix or ".jpg"
        out_path = self.upload_dir / f"{image_id}{suffix}"
        content = await upload_file.read()
        out_path.write_bytes(content)
        return image_id, out_path

    def output_path(self, image_id: str, leaf_idx: int) -> Path:
        """Build the output path for one leaf's Grad-CAM image.

        Args:
            image_id: Identifier generated when the original upload was saved.
            leaf_idx: Sequential leaf identifier from segmentation.

        Returns:
            Deterministic path where the overlay image should be written.
        """
        return self.output_dir / f"{image_id}_leaf_{leaf_idx}_gradcam.jpg"
