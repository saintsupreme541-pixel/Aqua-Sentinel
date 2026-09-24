"""Phase 1 persistence APIs: frames, frame detections, persistent targets.

Minimal read surface over the Survey → Frame → FrameDetection →
PersistentTarget hierarchy.  Existing endpoints are untouched; these extend
(never duplicate) the survey routes.  Target creation/association is an
explicit operator action (or a future Phase 2 algorithm) — Phase 1 never
fabricates targets from frame detections.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..pipeline.geolocate import normalize_status
from ..targets import run_association, target_history

router = APIRouter(prefix="/api", tags=["frames"])


def _frame_payload(img: dict[str, Any]) -> dict[str, Any]:
    """Shape one frame row (navigation fields pass through as NULL or value)."""
    return {
        "id": img["id"],
        "survey_id": img["survey_id"],
        "frame_index": img["frame_index"],
        "filename": img["filename"],
        "width": img["width"],
        "height": img["height"],
        "status": img["status"],
        "created_at": img["created_at"],
        "captured_at": img["captured_at"],
        "latitude": img["latitude"],
        "longitude": img["longitude"],
        "heading_deg": img["heading_deg"],
        "altitude_m": img["altitude_m"],
        "sonar_side": img["sonar_side"],
        "slant_range_m": img["slant_range_m"],
    }


@router.get("/surveys/{survey_id}/frames")
def list_frames(survey_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    return [_frame_payload(img) for img in db.get_images(survey_id)]


@router.get("/surveys/{survey_id}/frames/{frame_id}")
def frame_detail(survey_id: str, frame_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    img = db.get_image(frame_id)
    if not img or img["survey_id"] != survey_id:
        raise HTTPException(404, "frame not found in survey")
    return {**_frame_payload(img), "detections": db.get_detections(survey_id, image_id=frame_id)}


@router.get("/surveys/{survey_id}/frames/{frame_id}/detections")
def frame_detections(survey_id: str, frame_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    img = db.get_image(frame_id)
    if not img or img["survey_id"] != survey_id:
        raise HTTPException(404, "frame not found in survey")
    return db.get_detections(survey_id, image_id=frame_id)


# ---------------------------------------------------------------------------
# Persistent targets
# ---------------------------------------------------------------------------


class TargetCreate(BaseModel):
    """Explicit target creation — never auto-derived in Phase 1."""

    canonical_class: str = Field(min_length=1)
    status: Literal["active", "confirmed", "rejected", "review"] = "active"
    confidence: float | None = Field(default=None, ge=0, le=1)
    representative_detection_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    geolocation_status: Literal["known", "approximate", "unknown"] = "unknown"
    notes: str = ""


class TargetAssociate(BaseModel):
    """Link one frame detection to a target (same survey enforced in db layer)."""

    target_id: str | None = None
    detection_id: str


class AssociateRequest(BaseModel):
    """Body for the survey-level association run."""

    reset: bool = False


def _target_payload(t: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape one target row; ``detections`` are the survey's detections,
    used to count observations and list linked detection IDs."""
    evidence = None
    if t.get("geolocation_evidence_json"):
        try:
            evidence = json.loads(t["geolocation_evidence_json"])
        except ValueError:  # pragma: no cover — defensive
            evidence = None
    linked = [d for d in detections if d.get("target_id") == t["id"]]
    return {
        "id": t["id"],
        "survey_id": t["survey_id"],
        "canonical_class": t["canonical_class"],
        "status": t["status"],
        "confidence": t["confidence"],
        "first_seen_frame_id": t["first_seen_frame_id"],
        "last_seen_frame_id": t["last_seen_frame_id"],
        "representative_detection_id": t["representative_detection_id"],
        "latitude": t["latitude"],
        "longitude": t["longitude"],
        "geolocation_status": t["geolocation_status"],
        "location_status": normalize_status(t["geolocation_status"]),
        "geolocation_source": t.get("geolocation_source"),
        "geolocation_evidence": evidence,
        "n_observations": len(linked),
        "notes": t["notes"],
        "created_at": t["created_at"],
        "updated_at": t["updated_at"],
        "detection_ids": [d["id"] for d in linked],
    }


def _validate_representative(survey_id: str, detection_id: str | None) -> None:
    if detection_id is None:
        return
    row = db.get_detection_row(detection_id)
    if not row or row["survey_id"] != survey_id:
        raise HTTPException(400, "representative_detection_id must belong to this survey")


@router.get("/surveys/{survey_id}/targets")
def list_targets(survey_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    detections = db.get_detections(survey_id)
    return [_target_payload(t, detections) for t in db.get_targets(survey_id)]


@router.post("/surveys/{survey_id}/targets", status_code=201)
def create_target(survey_id: str, body: TargetCreate):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    _validate_representative(survey_id, body.representative_detection_id)

    # Optional convenience: seeding first/last_seen from the representative
    # detection's frame keeps the audit trail consistent without inventing data.
    first_seen = last_seen = None
    if body.representative_detection_id:
        det = db.get_detection_row(body.representative_detection_id)
        if det:
            first_seen = last_seen = det["image_id"]

    tid = db.create_target(
        survey_id,
        body.canonical_class,
        status=body.status,
        confidence=body.confidence,
        first_seen_frame_id=first_seen,
        last_seen_frame_id=last_seen,
        representative_detection_id=body.representative_detection_id,
        latitude=body.latitude,
        longitude=body.longitude,
        geolocation_status=body.geolocation_status,
        notes=body.notes,
    )
    if body.representative_detection_id:
        db.assign_detection_target(body.representative_detection_id, tid)
    t = db.get_target(tid)
    if t is None:  # pragma: no cover — just created in this request
        raise HTTPException(500, "target creation failed")
    return _target_payload(t, db.get_detections(survey_id))


@router.get("/surveys/{survey_id}/targets/{target_id}")
def target_detail(survey_id: str, target_id: str):
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    t = db.get_target(target_id)
    if not t or t["survey_id"] != survey_id:
        raise HTTPException(404, "target not found in survey")
    return _target_payload(t, db.get_detections(survey_id))


@router.patch("/surveys/{survey_id}/targets/{target_id}")
def update_target(survey_id: str, target_id: str, body: TargetCreate):
    """Update target fields (status/confidence/notes...). Fields set to null are ignored."""
    t = db.get_target(target_id)
    if not t or t["survey_id"] != survey_id:
        raise HTTPException(404, "target not found in survey")
    _validate_representative(survey_id, body.representative_detection_id)
    db.update_target(
        target_id,
        status=body.status,
        confidence=body.confidence,
        representative_detection_id=body.representative_detection_id,
        latitude=body.latitude,
        longitude=body.longitude,
        geolocation_status=body.geolocation_status,
        notes=body.notes,
    )
    t = db.get_target(target_id)
    if t is None:  # pragma: no cover — existence checked above
        raise HTTPException(404, "target not found in survey")
    return _target_payload(t, db.get_detections(survey_id))


@router.post("/surveys/{survey_id}/targets/associate")
def associate_detection(survey_id: str, body: TargetAssociate):
    """Link/unlink a frame detection to a target (target_id null = unlink)."""
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    det = db.get_detection_row(body.detection_id)
    if not det or det["survey_id"] != survey_id:
        raise HTTPException(404, "detection not found in survey")
    if body.target_id is not None:
        t = db.get_target(body.target_id)
        if not t or t["survey_id"] != survey_id:
            raise HTTPException(404, "target not found in survey")
    db.assign_detection_target(body.detection_id, body.target_id)
    return {"detection_id": body.detection_id, "target_id": body.target_id}


@router.delete("/surveys/{survey_id}/targets/{target_id}", status_code=204)
def delete_target(survey_id: str, target_id: str):
    t = db.get_target(target_id)
    if not t or t["survey_id"] != survey_id:
        raise HTTPException(404, "target not found in survey")
    db.delete_target(target_id)


# ---------------------------------------------------------------------------
# Phase 2: survey-level multi-frame association + target history
# ---------------------------------------------------------------------------


@router.post("/surveys/{survey_id}/associate")
def associate_survey(survey_id: str, body: AssociateRequest | None = None):
    """Run deterministic multi-frame association for the whole survey.

    Idempotent: existing valid links are reused; repeated calls do not
    duplicate targets.  ``{"reset": true}`` explicitly clears and rebuilds
    (for re-analyzed surveys).  Returns the run summary.
    """
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    try:
        summary = run_association(survey_id, reset=bool(body and body.reset))
    except Exception as e:  # transaction rolled back by db.transaction
        raise HTTPException(500, f"association failed: {e}") from e
    return summary


@router.get("/surveys/{survey_id}/targets/{target_id}/history")
def target_history_route(survey_id: str, target_id: str):
    """Target identity + full detection history in deterministic frame order."""
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    h = target_history(survey_id, target_id)
    if not h:
        raise HTTPException(404, "target not found in survey")
    return h


# ---------------------------------------------------------------------------
# Phase 3: survey-level spatial summary — authoritative data for the map.
# Coordinates appear ONLY when genuinely available; nothing is fabricated.
# ---------------------------------------------------------------------------


@router.get("/surveys/{survey_id}/spatial")
def survey_spatial(survey_id: str):
    """Spatial summary: frame positions, track, target locations, availability.

    - ``track`` preserves frame order and contains ONLY observed frame
      positions (never interpolated through gaps).
    - ``targets`` carries the deterministic representative location with its
      provenance; targets without a reliable fix are listed with
      ``geolocation_status: "unknown"`` and null coordinates.
    """
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")

    frames = db.get_images(survey_id)  # already frame_index-ordered (db.get_images)
    frames_with_pos: list[dict[str, Any]] = []
    for im in frames:
        if im.get("latitude") is not None and im.get("longitude") is not None:
            frames_with_pos.append(
                {
                    "frame_id": im["id"],
                    "frame_index": im.get("frame_index"),
                    "filename": im["filename"],
                    "latitude": im["latitude"],
                    "longitude": im["longitude"],
                    "heading_deg": im.get("heading_deg"),
                    "captured_at": im.get("captured_at"),
                    "position_source": im.get("nav_source") or "observed",
                    "sync_method": im.get("nav_sync_method"),
                }
            )
    track = [[f["longitude"], f["latitude"]] for f in frames_with_pos]

    # Geolocation v2: the genuine uploaded navigation track (§5) — separate
    # from frame positions; rendered as its own MapLibre line source.
    nav_records = db.get_nav_records(survey_id)
    nav_coordinates = [[r["longitude"], r["latitude"]] for r in nav_records if r.get("longitude") is not None]
    nav_track = {
        "available": len(nav_coordinates) >= 2,
        "record_count": len(nav_records),
        "coordinates": nav_coordinates if len(nav_coordinates) >= 2 else None,
        "note": "genuine uploaded navigation in timestamp order (§5) — never generated for display",
    }

    detections = db.get_detections(survey_id)
    targets = [_target_payload(t, detections) for t in db.get_targets(survey_id)]
    located_targets = [t for t in targets if t["latitude"] is not None and t["longitude"] is not None]

    lats = [f["latitude"] for f in frames_with_pos] + [t["latitude"] for t in located_targets]
    lons = [f["longitude"] for f in frames_with_pos] + [t["longitude"] for t in located_targets]
    bounds = (
        [
            [min(lons), min(lats)],
            [max(lons), max(lats)],
        ]
        if lats and lons
        else None
    )

    return {
        "survey_id": survey_id,
        "frames_total": len(frames),
        "frames_with_location": len(frames_with_pos),
        "frames_without_location": len(frames) - len(frames_with_pos),
        "targets_total": len(targets),
        "targets_with_location": len(located_targets),
        "targets_without_location": len(targets) - len(located_targets),
        "bounds": bounds,
        "nav_track": nav_track,
        "track": {
            "available": len(track) >= 2,
            "point_count": len(track),
            "note": (
                "observed frame positions in frame order; gaps are NOT interpolated"
                if len(track) >= 2
                else "insufficient observed frame positions for a track"
            ),
            "points": frames_with_pos,
            "coordinates": track,
        },
        "frame_positions": frames_with_pos,
        "targets": targets,
    }
