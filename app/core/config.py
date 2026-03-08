"""Central application settings loaded from environment variables.

This module provides a cached `Settings` instance used by both backend
runtime and scripts.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed configuration for API, model, and storage paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )
    app_name: str = Field(default="Plant Disease Detector")
    debug: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    confidence_threshold: float = Field(default=0.70)
    max_leaves: int = Field(default=5)
    model_name: str = Field(default="mobilenet")
    model_weights_path: str = Field(default="ml/weights/mobilenet_best.pt")
    class_names_path: str = Field(default="ml/weights/classes.json")
    target_image_size: int = Field(default=384)
    strict_model_loading: bool = Field(default=True)

    upload_dir: str = Field(default="app/static/uploads")
    output_dir: str = Field(default="app/static/outputs")
    analytics_log: str = Field(default="logs/analytics.jsonl")
    app_log: str = Field(default="logs/app.log")

    def ensure_dirs(self) -> None:
        """Create configured output/log directories if they do not exist."""
        for path in [self.upload_dir, self.output_dir, Path(self.analytics_log).parent, Path(self.app_log).parent]:
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached settings object."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
