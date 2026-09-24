from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import db

router = APIRouter(prefix="/api/detections", tags=["detections"])


@router.get("/{detection_id}")
def detection_detail(detection_id: str):
    """One detection by ID, with its survey/image scope.

    Reads the row directly instead of scanning all surveys' detections; the
    response carries ``survey_id``/``image_id``/``target_id`` so a client can
    never mistake a Survey-A detection for Survey-B data.
    """
    row = db.get_detection_row(detection_id)
    if not row:
        raise HTTPException(404, "detection not found")
    return row
