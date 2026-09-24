"""Filesystem layout for survey artifacts.

Layout under the storage root::

    raw/        original uploaded files (immutable)
    processed/  <survey>/<image>/processed.png      preprocessed image
    masks/      <survey>/<image>/mask_<detid>.png   segmentation masks
    overlays/   <survey>/<image>/overlay.png        detection overlay (for the Lab)
    reports/    <survey>/<format>/<report>.<ext>
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import settings

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)


def safe_filename(name: str) -> str:
    """Sanitize an uploaded filename for on-disk storage.

    Unsafe characters are *replaced* (not stripped) so two distinct uploads
    can never silently collapse into the same file name.  Falls back to a
    generic name when nothing safe remains.
    """
    cleaned = _safe(name).strip("._") or "image.png"
    return cleaned


def raw_dir(survey_id: str) -> Path:
    p = settings.storage_dir / "raw" / _safe(survey_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def image_dir(survey_id: str, image_id: str, kind: str) -> Path:
    p = settings.storage_dir / kind / _safe(survey_id) / _safe(image_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir(survey_id: str) -> Path:
    p = settings.storage_dir / "reports" / _safe(survey_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_relative(rel: str) -> str:
    """Sanitize a storage-relative path for media serving."""
    rel = rel.replace("\\", "/")
    if ".." in rel.split("/"):
        raise ValueError("unsafe path")
    return rel.lstrip("/")


def to_media_url(rel: str) -> str:
    return f"/api/media/{safe_relative(rel)}"
