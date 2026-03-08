"""File storage helper for uploaded images and generated outputs."""

import uuid
from pathlib import Path

from fastapi import UploadFile


class StorageService:
    """Handles persistence paths for inputs and Grad-CAM output images."""

    def __init__(self, upload_dir: str, output_dir: str):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload_file: UploadFile) -> tuple[str, Path]:
        """Save uploaded file and return generated image_id with absolute path."""
        image_id = uuid.uuid4().hex
        suffix = Path(upload_file.filename or "upload.jpg").suffix or ".jpg"
        out_path = self.upload_dir / f"{image_id}{suffix}"
        content = await upload_file.read()
        out_path.write_bytes(content)
        return image_id, out_path

    def output_path(self, image_id: str, leaf_idx: int) -> Path:
        """Build deterministic output path for a leaf-level Grad-CAM image."""
        return self.output_dir / f"{image_id}_leaf_{leaf_idx}_gradcam.jpg"
