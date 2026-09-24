from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import db
from ..config import settings
from ..schemas import SurveyMeta

router = APIRouter(prefix="/api/surveys", tags=["surveys"])


def _raw_media_url(image_row: dict) -> str:
    """Map a stored raw_path (absolute) to its public /api/media URL."""
    from ..storage import to_media_url

    rel = (image_row.get("raw_path") or "").replace(str(settings.storage_dir), "").replace("\\", "/").lstrip("/")
    return to_media_url(rel)


def _survey_card(s: dict, det_count: int) -> dict:
    images = db.get_images(s["id"])
    job = db.latest_job(s["id"])
    return {
        "id": s["id"],
        "name": s["name"],
        "description": s.get("description", ""),
        "created_at": s["created_at"],
        "image_count": len(images),
        "detection_count": det_count,
        "job_status": job["status"] if job else None,
        "job_id": job["id"] if job else None,
        "meta": json.loads(s["meta_json"]),
    }


@router.get("")
def list_surveys():
    return [_survey_card(s, len(db.get_detections(s["id"]))) for s in db.list_surveys()]


@router.post("")
async def create_survey(
    files: list[UploadFile] = File(...),
    meta: str = Form("{}"),
    image_meta: str = Form("{}"),
):
    try:
        survey_meta = SurveyMeta(**json.loads(meta))
    except Exception as e:
        raise HTTPException(400, f"invalid survey metadata: {e}") from e
    try:
        per_image: dict[str, dict[str, Any]] = json.loads(image_meta)
    except Exception as e:
        raise HTTPException(400, f"invalid per-image metadata: {e}") from e

    entries = [(f.filename or "image.png", await f.read()) for f in files if f.filename]
    if not entries:
        raise HTTPException(400, "no image files provided")

    from ..service import create_survey_with_files

    try:
        sid, job_id = create_survey_with_files(entries, survey_meta.model_dump(), per_image)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"survey_id": sid, "job_id": job_id, "image_count": len(entries)}


@router.post("/{survey_id}/reanalyze")
def reanalyze_survey(survey_id: str):
    """Re-run the analysis job for an existing survey (geolocation v2).

    Needed after a navigation track is uploaded: frames are synchronized to
    the track during analysis, so the job must re-run for the new provenance
    to take effect.  Only already-uploaded images are re-analyzed — nothing
    new is fabricated, and the job pipeline is identical.
    """
    s = db.get_survey(survey_id)
    if not s:
        raise HTTPException(404, "survey not found")
    if not db.get_images(survey_id):
        raise HTTPException(400, "survey has no images to analyze")
    job_id = db.create_job(survey_id)
    from .. import jobs

    jobs.submit_analysis(survey_id, job_id)
    return {"survey_id": survey_id, "job_id": job_id}


@router.get("/{survey_id}")
def survey_detail(survey_id: str):
    s = db.get_survey(survey_id)
    if not s:
        raise HTTPException(404, "survey not found")
    images = db.get_images(survey_id)
    dets = db.get_detections(survey_id)
    return {
        **_survey_card(s, len(dets)),
        "images": [
            {
                "id": im["id"],
                "filename": im["filename"],
                "width": im["width"],
                "height": im["height"],
                "status": im["status"],
                "meta": json.loads(im["meta_json"]),
                "raw_url": _raw_media_url(im),
            }
            for im in images
        ],
    }


@router.get("/{survey_id}/detections")
def survey_detections(survey_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    return db.get_detections(survey_id)


@router.get("/{survey_id}/geojson")
def survey_geojson(survey_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    from ..reports import collect_survey_data
    from ..reports.geojson_report import render

    return json.loads(render(collect_survey_data(survey_id)))


@router.get("/{survey_id}/images/{image_id}")
def image_result(survey_id: str, image_id: str):
    """Full per-image result + artifact URLs for the Sonar Intelligence Lab."""
    img = db.get_image(image_id)
    if not img or img["survey_id"] != survey_id:
        raise HTTPException(404, "image not found in survey")
    result = db.get_image_result(image_id) or {}
    from ..storage import to_media_url

    return {
        "image_id": image_id,
        "survey_id": survey_id,
        "filename": img["filename"],
        "width": img["width"],
        "height": img["height"],
        "meta": json.loads(img["meta_json"]),
        "raw_url": _raw_media_url(img),
        "processed_url": to_media_url(result.get("processed_path", "")) if result.get("processed_path") else None,
        "overlay_url": f"/api/media/overlays/{survey_id}/{image_id}/overlay.png",
        "quality": result.get("quality"),
        "preprocess_params": result.get("preprocess_params", {}),
        "backends": result.get("backends", {}),
        "backend_notes": result.get("backend_notes", []),
        "status": img["status"],
    }
