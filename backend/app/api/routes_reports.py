from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import db
from ..config import settings
from ..reports import collect_survey_data
from ..reports.csv_report import render as render_csv
from ..reports.geojson_report import render as render_geojson
from ..reports.json_report import render as render_json
from ..reports.pdf_report import render as render_pdf

router = APIRouter(prefix="/api", tags=["reports"])

#: supported formats → their renderer; the format name doubles as the file
#: extension and each renderer receives exactly what it needs (see _build).
RENDERERS: dict[str, Callable[..., Any]] = {
    "csv": render_csv,
    "json": render_json,
    "geojson": render_geojson,
    "pdf": render_pdf,
}


@router.post("/surveys/{survey_id}/reports")
def generate_reports(survey_id: str, formats: list[str]):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    unknown = [f for f in formats if f not in RENDERERS]
    if unknown:
        raise HTTPException(400, f"unsupported formats: {unknown}")
    data = collect_survey_data(survey_id)
    created = []
    for fmt in formats:
        rid, rel = _build(fmt, survey_id, data)
        created.append(rid)
    return {"survey_id": survey_id, "report_ids": created}


def _build(fmt: str, survey_id: str, data: dict) -> tuple[str, str]:
    out_dir = settings.storage_dir / "reports" / survey_id
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"aqua-sentinel-{survey_id}.{fmt}"

    renderer = RENDERERS[fmt]
    if fmt == "csv":
        base.write_text(renderer(data["detections"]), encoding="utf-8")
    elif fmt == "pdf":
        renderer(data, base)
    else:
        base.write_text(renderer(data), encoding="utf-8")

    size = base.stat().st_size
    rid = db.create_report(survey_id, fmt, str(base), size)
    return rid, str(base)


@router.get("/surveys/{survey_id}/reports")
def list_reports(survey_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    return db.list_reports(survey_id)


@router.get("/reports/{report_id}/download")
def download_report(report_id: str):
    rep = db.get_report(report_id)
    if not rep:
        raise HTTPException(404, "report not found")
    p = Path(rep["path"])
    if not p.exists():
        raise HTTPException(404, "report file missing")
    return FileResponse(p, filename=p.name, media_type=_media_type(rep["format"]))


def _media_type(fmt: str) -> str:
    return {
        "csv": "text/csv",
        "json": "application/json",
        "geojson": "application/geo+json",
        "pdf": "application/pdf",
    }.get(fmt, "application/octet-stream")
