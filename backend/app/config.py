"""Application configuration loaded from environment variables with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    storage_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("AQUA_STORAGE_DIR", _PROJECT_ROOT / "data" / "runtime"))
    )
    db_path: Path | None = field(
        default_factory=lambda: (lambda v: Path(v) if v else None)(os.environ.get("AQUA_DB_PATH", ""))
    )
    model_registry_path: Path = field(
        default_factory=lambda: Path(os.environ.get("AQUA_MODEL_REGISTRY", _PROJECT_ROOT / "models" / "registry.json"))
    )
    preprocess_preset: str = os.environ.get("AQUA_PREPROCESS_PRESET", "light")
    max_workers: int = int(os.environ.get("AQUA_MAX_WORKERS", "2"))
    max_upload_bytes: int = int(os.environ.get("AQUA_MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))
    host: str = os.environ.get("AQUA_HOST", "0.0.0.0")
    port: int = int(os.environ.get("AQUA_PORT", "8000"))
    # Fusion / decision thresholds (documented in docs/metrics.md)
    fusion_gain: float = float(os.environ.get("AQUA_FUSION_GAIN", "3.2"))
    fusion_intercept: float = float(os.environ.get("AQUA_FUSION_INTERCEPT", "-0.4"))
    confirm_threshold: float = float(os.environ.get("AQUA_CONFIRM_THRESHOLD", "0.63"))
    review_threshold: float = float(os.environ.get("AQUA_REVIEW_THRESHOLD", "0.45"))
    consistency_distance_m: float = float(os.environ.get("AQUA_CONSISTENCY_DISTANCE_M", "25.0"))
    sample_data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("AQUA_SAMPLE_DATA", _PROJECT_ROOT / "data" / "samples"))
    )

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = self.storage_dir / "aqua.db"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("raw", "processed", "masks", "overlays", "reports", "jobs"):
            (self.storage_dir / sub).mkdir(parents=True, exist_ok=True)


settings = Settings()
