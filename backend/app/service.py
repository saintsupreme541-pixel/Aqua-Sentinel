"""Survey creation service shared by the upload route and sample import."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import io
from PIL import Image

from . import db, jobs
from .schemas import ImageMeta, SurveyMeta
from .storage import raw_dir, safe_filename


def extract_exif_metadata(content: bytes) -> dict[str, Any]:
    """Extract embedded EXIF/TIFF GPS and capture timestamp metadata from image bytes."""
    extracted: dict[str, Any] = {}
    try:
        with Image.open(io.BytesIO(content)) as img:
            exif = img.getexif()
            if not exif:
                return extracted

            gps_info = exif.get_ifd(0x8825)
            if gps_info:
                lat_ref = gps_info.get(1)
                lat_val = gps_info.get(2)
                lon_ref = gps_info.get(3)
                lon_val = gps_info.get(4)
                alt_val = gps_info.get(6)

                if lat_val and lat_ref:
                    try:
                        lat_deg = float(lat_val[0]) + float(lat_val[1]) / 60.0 + float(lat_val[2]) / 3600.0
                        if str(lat_ref).strip().upper() == "S":
                            lat_deg = -lat_deg
                        if -90.0 <= lat_deg <= 90.0:
                            extracted["lat"] = round(lat_deg, 7)
                    except Exception:
                        pass

                if lon_val and lon_ref:
                    try:
                        lon_deg = float(lon_val[0]) + float(lon_val[1]) / 60.0 + float(lon_val[2]) / 3600.0
                        if str(lon_ref).strip().upper() == "W":
                            lon_deg = -lon_deg
                        if -180.0 <= lon_deg <= 180.0:
                            extracted["lon"] = round(lon_deg, 7)
                    except Exception:
                        pass

                if alt_val:
                    try:
                        alt = float(alt_val)
                        if alt > 0:
                            extracted["altitude_m"] = round(alt, 2)
                    except Exception:
                        pass

            dt_str = exif.get(36867) or exif.get(306)
            if dt_str:
                extracted["captured_at"] = str(dt_str).strip()
    except Exception:
        pass
    return extracted


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
        exif_meta = extract_exif_metadata(content)
        overrides = per_image_overrides.get(filename, {})
        # Merge: EXIF metadata provides defaults, overridden by survey defaults and per-image overrides
        merged_meta = {**exif_meta, **{k: v for k, v in meta_model.model_dump().items() if v is not None}, **overrides, "filename": filename}
        imeta = ImageMeta(**merged_meta)
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
