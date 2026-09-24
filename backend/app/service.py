"""Survey creation service shared by the upload route and sample import."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import db, jobs
from .schemas import ImageMeta, SurveyMeta
from .storage import raw_dir, safe_filename


def create_survey_with_files(
    entries: list[tuple[str, bytes]],
    survey_meta: dict[str, Any],
    per_image_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Persist raw files + metadata, then kick off the analysis job.

    ``entries``: list of (filename, bytes).  Images are validated client-side
    here as a second gate (both upload and sample import use this path).
    """
    per_image_overrides = per_image_overrides or {}
    validated: list[tuple[str, bytes, int, int]] = []
    for filename, content in entries:
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"'{filename}' is not a decodable image")
        h, w = img.shape
        validated.append((filename, content, w, h))

    meta_model = SurveyMeta(**survey_meta)
    sid = db.create_survey(meta_model.name, meta_model.description, meta_model.model_dump(), len(validated))

    raw = raw_dir(sid)
    image_ids: list[str] = []
    for filename, content, w, h in validated:
        target = raw / safe_filename(filename)
        target.write_bytes(content)
        overrides = per_image_overrides.get(filename, {})
        imeta = ImageMeta(**{**meta_model.model_dump(), **overrides, "filename": filename})
        nav = _frame_nav_columns(imeta.model_dump())
        image_ids.append(db.add_image(sid, filename, str(target), imeta.model_dump(), w, h, **nav))

    job_id = db.create_job(sid)
    jobs.submit_analysis(sid, job_id)
    return sid, job_id


def _frame_nav_columns(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract frame-level navigation columns from provided metadata.

    Only values the operator actually supplied are stored; missing fields
    stay NULL in the database (the system never fabricates GPS, heading,
    altitude, side, slant range or capture time).
    """
    side = meta.get("side") or "unknown"
    return {
        "captured_at": meta.get("captured_at"),
        "latitude": meta.get("lat"),
        "longitude": meta.get("lon"),
        "heading_deg": meta.get("heading_deg"),
        "altitude_m": meta.get("altitude_m"),
        "sonar_side": side if side in ("port", "starboard") else None,
        "slant_range_m": meta.get("range_m"),
    }
