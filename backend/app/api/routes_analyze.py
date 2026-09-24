"""Single-image analysis endpoints (spec §17/§18).

``/api/analyze`` is the primary synchronous endpoint: it accepts a sonar
image (or up to 8 frames for multi-frame consistency), optional metadata
(GPS/heading/altitude/range) and runs the full stage pipeline (quality →
preprocess → detect → segment → classify → shadow → physics → anomaly →
fusion) **in-memory, without persistence** — the survey/jobs endpoints
remain the persisted workflow.  ``/api/detect`` is the lightweight
detection-only variant.  ``/api/metadata`` describes the optional sonar
metadata fields and their units.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import analysis_service
from ..config import settings

router = APIRouter(prefix="/api", tags=["analyze"])

_MAX_FRAMES = 8
_ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _validate(file: UploadFile, content: bytes) -> None:
    name = (file.filename or "").lower()
    if not name:
        raise HTTPException(400, "file has no name")
    if not name.endswith(_ALLOWED_EXT):
        raise HTTPException(400, f"unsupported file type: {name}")
    if len(content) == 0:
        raise HTTPException(400, "empty file")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"file exceeds {settings.max_upload_bytes // (1024 * 1024)} MB limit")


def _parse_meta(raw: str | None) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(422, "metadata must be valid JSON") from e
    if not isinstance(meta, dict):
        raise HTTPException(422, "metadata must be a JSON object")
    return meta


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Primary sonar image"),
    metadata: str | None = Form(None, description="Optional JSON metadata object"),
    preset: str = Form("light"),
    frames: list[UploadFile] = File(None, description="Optional extra frames for multi-frame consistency"),
):
    contents: list[tuple[str, bytes]] = []
    primary = await file.read()
    _validate(file, primary)
    contents.append((file.filename or "sonar.png", primary))

    extra = [f for f in (frames or []) if f is not None and f.filename]
    if len(contents) + len(extra) > _MAX_FRAMES:
        raise HTTPException(413, f"too many frames (max {_MAX_FRAMES})")
    for f in extra:
        c = await f.read()
        _validate(f, c)
        contents.append((f.filename or "frame.png", c))

    meta = _parse_meta(metadata)
    meta.setdefault("sonar_type", "sss")
    try:
        return analysis_service.analyze_images(contents, meta, preset=preset)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/detect")
async def detect_only(
    file: UploadFile = File(..., description="Sonar image"),
    metadata: str | None = Form(None),
):
    """Detection-only pass (quality + preprocess + detect). No fusion."""
    content = await file.read()
    _validate(file, content)
    meta = _parse_meta(metadata)
    meta.setdefault("sonar_type", "sss")
    try:
        result = analysis_service.analyze_images([(file.filename or "sonar.png", content)], meta)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "analysis_id": result["analysis_id"],
        "backends": result["backends"],
        "detections": [
            {
                "id": d["id"],
                "class": d["class"],
                "confidence": d["yolo_confidence"],
                "bbox": d["bbox"],
            }
            for d in result["detections"]
        ],
    }


@router.get("/metadata")
async def metadata_fields():
    """Document the optional sonar metadata contract (no values invented)."""
    return {
        "fields": {
            "lat": {"type": "float", "unit": "degrees WGS84", "required": False},
            "lon": {"type": "float", "unit": "degrees WGS84", "required": False},
            "heading_deg": {"type": "float", "unit": "degrees true", "required": False},
            "altitude_m": {"type": "float", "unit": "m (sonar height above seabed)", "required": False},
            "range_m": {"type": "float", "unit": "m (max slant range of the image)", "required": False},
            "sonar_type": {"type": "enum", "values": ["sss", "fls", "other", "unknown"], "required": False},
            "side": {"type": "enum", "values": ["port", "starboard", "unknown"], "required": False},
            "frequency_khz": {"type": "float", "unit": "kHz", "required": False},
            "pixel_resolution_m": {"type": "float", "unit": "m/px (overrides derived scale)", "required": False},
        },
        "policy": "Missing fields degrade gracefully — the system never fabricates coordinates or physical values.",
    }
