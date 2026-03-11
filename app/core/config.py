"""Environment-backed configuration for the deployed application.

The project uses a single settings object for runtime paths, model-selection
defaults, confidence thresholds, and logging destinations. Centralizing these
values makes the API easier to reason about and reduces the chance of route or
service modules hard-coding inconsistent behavior.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration loaded from ``.env``.

    The defaults are chosen to match the repository layout and the MobileNetV2
    deployment model described in the project documentation.
    """

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
        """Create directories required by runtime storage and logging.

        Side Effects:
            Creates upload, output, and log directories when missing.
        """
        required_paths = [
            self.upload_dir,
            self.output_dir,
            Path(self.analytics_log).parent,
            Path(self.app_log).parent,
        ]
        for path in required_paths:
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide settings instance.

    Returns:
        The lazily created :class:`Settings` object shared across the process.
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings
