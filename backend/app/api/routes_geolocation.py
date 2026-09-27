"""Geolocation v2 APIs — navigation ingestion, diagnostics, manual fixes.

Provenance-first geolocation endpoints (spec §1–§19):

- ``POST /surveys/{id}/navigation``  — upload a genuine navigation track
  (CSV / GeoCSV / GeoJSON / GPX).  Records are validated; invalid rows are
  rejected WITH diagnostics, never repaired, never stored as usable nav.
- ``GET  /surveys/{id}/navigation``  — the stored track + ingestion stats.
- ``DELETE /surveys/{id}/navigation`` — remove the track (explicit action).
- ``GET  /surveys/{id}/geolocation/diagnostics`` — per-frame navigation
  sync, provenance availability and position-quality report for the Lab.
- ``POST /detections/{id}/geolocation/manual`` — explicit user georeference
  (status MANUAL, source user_supplied; never presented as GNSS).

Every coordinate leaving these endpoints carries a ``location_status`` from
the v2 state model (VERIFIED/DERIVED/MANUAL/VESSEL_ONLY/UNAVAILABLE/INVALID).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .. import db
from .. import navigation as nav
from ..pipeline import geolocate

router = APIRouter(prefix="/api", tags=["geolocation"])


def _dump_json(value: Any) -> str:
    """Serialize provenance for a TEXT column (empty string when absent)."""
    return json.dumps(value) if value is not None else ""


@router.post("/surveys/{survey_id}/navigation")
async def upload_navigation(
    survey_id: str,
    file: UploadFile = File(..., description="Navigation track: CSV, GeoCSV, GeoJSON or GPX"),
    crs: str | None = Form(None, description="Source CRS (e.g. EPSG:32644 or UTM44N); default WGS84"),
) -> dict[str, Any]:
    """Ingest a genuine navigation track for a survey (§2/§3).

    The whole file is validated BEFORE anything is stored: rows with
    invalid coordinates/timestamps are rejected with explicit reasons,
    duplicate/impossible-jump records are dropped and reported.  Only the
    surviving valid records become the survey track.
    """
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty navigation file")

    result = nav.ingest_navigation(file.filename or "", content, file_crs=crs)
    records = result["records"]
    if not records:
        raise HTTPException(
            422,
            {
                "message": "no valid navigation records in file",
                "warnings": result["warnings"],
                "rejected": result["rejected"][:20],
                "crs": result["crs"],
            },
        )

    db.replace_nav_records(survey_id, records)
    return {
        "survey_id": survey_id,
        "stored": len(records),
        "stats": result["stats"],
        "rejected": result["rejected"][:50],
        "warnings": result["warnings"],
        "crs": result["crs"],
        "span": result["span"],
    }


@router.get("/surveys/{survey_id}/navigation")
def get_navigation(survey_id: str) -> dict[str, Any]:
    """The stored navigation track + ingestion diagnostics (§3/§5)."""
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    records = db.get_nav_records(survey_id)
    stats = db.nav_stats(survey_id)
    coordinates = [[r["longitude"], r["latitude"]] for r in records if r.get("longitude") is not None]
    span = None
    if records:
        dated = [r["ts"] for r in records if r.get("ts")]
        if dated:
            span = {"start": min(dated), "end": max(dated)}
    return {
        "survey_id": survey_id,
        "available": len(records) >= 2,
        "record_count": len(records),
        "stats": stats,
        "span": span,
        "note": (
            "genuine uploaded navigation in timestamp order — never interpolated for display"
            if len(records) >= 2
            else "insufficient navigation records for a track"
        ),
        "geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates} if len(coordinates) >= 2 else None,
            "properties": {"survey_id": survey_id, "record_count": len(records), "kind": "survey_track"},
        },
        "records": [
            {
                "id": r["id"],
                "ts": r.get("ts"),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "heading_deg": r.get("heading_deg"),
                "altitude_m": r.get("altitude_m"),
                "accuracy_m": r.get("accuracy_m"),
                "source": r.get("source"),
            }
            for r in records
        ],
    }


@router.delete("/surveys/{survey_id}/navigation")
def delete_navigation(survey_id: str) -> dict[str, Any]:
    """Remove the survey's navigation track (explicit operator action)."""
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")
    n = db.nav_stats(survey_id)["total"]
    db.replace_nav_records(survey_id, [])
    return {"survey_id": survey_id, "removed": n}


@router.get("/surveys/{survey_id}/geolocation/diagnostics")
def geolocation_diagnostics(survey_id: str) -> dict[str, Any]:
    """Position-quality report for the Sonar Intelligence Lab (§14).

    Honest availability of every navigation input, the frame-level sync
    method, and per-target status — missing items are shown as missing,
    never as zeros (§14/§15).
    """
    if not db.get_survey(survey_id):
        raise HTTPException(404, "survey not found")

    nav_records = db.get_nav_records(survey_id)
    stats = db.nav_stats(survey_id)
    frames = db.get_images(survey_id)

    dated = [r for r in nav_records if r.get("ts")]
    has_track = len(nav_records) >= 2
    has_heading = any(r.get("heading_deg") is not None for r in nav_records)
    has_towfish = any(
        any(r.get(k) is not None for k in ("towfish_x", "towfish_y", "towfish_z", "layback")) for r in nav_records
    )

    def _flag(ok: bool) -> str:
        return "ok" if ok else "missing"

    frames_diag = []
    for im in frames:
        synced = im.get("nav_sync_method")
        frames_diag.append(
            {
                "frame_id": im["id"],
                "filename": im["filename"],
                "captured_at": im.get("captured_at"),
                "frame_position": {
                    "latitude": im.get("latitude"),
                    "longitude": im.get("longitude"),
                    "position_source": im.get("nav_source")
                    or ("frame_metadata" if im.get("latitude") is not None else None),
                },
                "sync_method": synced,
                "in_track_span": bool(
                    synced
                    and dated
                    and im.get("captured_at")
                    and (min(r["ts"] for r in dated) <= im["captured_at"] <= max(r["ts"] for r in dated))
                ),
            }
        )

    targets = []
    for t in db.get_targets(survey_id):
        status = geolocate.normalize_status(t.get("geolocation_status"))
        targets.append(
            {
                "target_id": t["id"],
                "canonical_class": t["canonical_class"],
                "location_status": status,
                "latitude": t.get("latitude"),
                "longitude": t.get("longitude"),
                "uncertainty_m": _target_uncertainty(t),
                "position_source": t.get("geolocation_source"),
            }
        )

    return {
        "survey_id": survey_id,
        "navigation": {
            "gnss_data": _flag(stats["valid"] > 0),
            "survey_track": _flag(has_track),
            "timestamp": _flag(bool(dated)),
            "crs": "ok",  # stored track is always normalized to EPSG:4326
            "heading": _flag(has_heading),
            "towfish_data": _flag(has_towfish),
            "layback": _flag(any(r.get("layback") is not None for r in nav_records)),
            "sonar_geometry": _flag(
                any(im.get("altitude_m") is not None and im.get("slant_range_m") is not None for im in frames)
                or any(im.get("altitude_m") is not None for im in frames)
            ),
            "stats": stats,
            "record_count": len(nav_records),
        },
        "position_quality": {
            "targets": targets,
            "targets_total": len(targets),
            "located": sum(1 for t in targets if t["latitude"] is not None),
            "status_counts": _status_counts(targets),
        },
        "frames": frames_diag,
    }


def _target_uncertainty(t: dict[str, Any]) -> float | None:
    ev = t.get("geolocation_evidence_json")
    if not ev:
        return None
    try:
        data = json.loads(ev)
    except ValueError:  # pragma: no cover — defensive
        return None
    det_id = data.get("from_detection_id")
    if not det_id:
        return None
    det = db.get_detection_row(det_id)
    if not det:
        return None
    return (det.get("geolocation") or {}).get("uncertainty_m")


def _status_counts(targets: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in targets:
        s = t["location_status"]
        counts[s] = counts.get(s, 0) + 1
    return counts


class ManualGeoref(BaseModel):
    """Explicit user-created georeference (§16)."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    note: str = Field(default="", max_length=500)


# A target georeference carries exactly the same fields as a detection one.
TargetManualGeoref = ManualGeoref


@router.post("/detections/{detection_id}/geolocation/manual")
def manual_georeference_detection(detection_id: str, body: ManualGeoref) -> dict[str, Any]:
    """Attach a MANUAL geolocation to one frame detection (§16).

    Status is MANUAL / source user_supplied — never presented as GNSS or
    geometry-derived.  The operator's coordinate replaces nothing silently:
    it is recorded with its provenance and visibly labelled.
    """
    det = db.get_detection_row(detection_id)
    if not det:
        raise HTTPException(404, "detection not found")
    geo = geolocate.manual_geolocation(
        body.latitude, body.longitude, note=body.note, reference_frame_id=det.get("image_id")
    )
    det["geolocation"] = geo
    db.update_detections([det])
    return {"detection_id": detection_id, "geolocation": geo}


@router.post("/surveys/{survey_id}/targets/{target_id}/geolocation/manual")
def manual_georeference_target(survey_id: str, target_id: str, body: TargetManualGeoref) -> dict[str, Any]:
    """Attach a MANUAL geolocation to a persistent target (§16)."""
    t = db.get_target(target_id)
    if not t or t["survey_id"] != survey_id:
        raise HTTPException(404, "target not found in survey")
    geo = geolocate.manual_geolocation(body.latitude, body.longitude, note=body.note)
    db.update_target(
        target_id,
        latitude=body.latitude,
        longitude=body.longitude,
        geolocation_status="MANUAL",
        geolocation_source="user_supplied",
        geolocation_evidence_json=_dump_json(geo.get("provenance")),
    )
    t = db.get_target(target_id)
    assert t is not None  # just updated
    return {"target_id": target_id, "geolocation": geo, "target": _target_view(t)}


def _target_view(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": t["id"],
        "latitude": t.get("latitude"),
        "longitude": t.get("longitude"),
        "geolocation_status": t.get("geolocation_status"),
        "geolocation_source": t.get("geolocation_source"),
    }


class GeolocationCalculateRequest(BaseModel):
    """Interactive Sonar Geolocation calculation request."""

    latitude: float = Field(..., ge=-90, le=90, description="Vessel / Sensor latitude (deg)")
    longitude: float = Field(..., ge=-180, le=180, description="Vessel / Sensor longitude (deg)")
    heading_deg: float = Field(..., ge=0, le=360, description="Vessel heading (deg 0-360)")
    altitude_m: float = Field(..., gt=0, description="Sensor altitude above seabed (m)")
    side: str = Field(..., description="Sonar side: 'port' or 'starboard'")
    slant_range_m: float | None = Field(None, gt=0, description="Direct slant range to target (m)")
    row: float | None = Field(None, ge=0, description="Image pixel row of detection")
    image_height: int = Field(default=0, ge=0, description="Image pixel height")
    range_m: float | None = Field(None, gt=0, description="Maximum sonar slant range (m)")
    layback_m: float | None = Field(None, ge=0, description="Towfish layback offset behind vessel (m)")
    roll_deg: float | None = Field(None, ge=-45, le=45, description="Transducer roll angle (deg)")
    pitch_deg: float | None = Field(None, ge=-45, le=45, description="Transducer pitch angle (deg)")
    accuracy_m: float | None = Field(None, gt=0, description="GNSS / Sensor accuracy (m)")


@router.post("/geolocation/calculate")
def calculate_geolocation(body: GeolocationCalculateRequest) -> dict[str, Any]:
    """Interactive Sonar Geolocation Calculator.

    Derives object coordinates, ground range, bearing, 2-sigma uncertainty ellipse,
    and GeoJSON uncertainty polygon from observed vessel position, heading, altitude,
    side, slant range, and optional sensor parameters (layback, roll, pitch, accuracy).
    """
    meta = {
        "lat": body.latitude,
        "lon": body.longitude,
        "heading_deg": body.heading_deg,
        "altitude_m": body.altitude_m,
        "side": body.side,
        "range_m": body.range_m,
        "layback_m": body.layback_m,
        "roll_deg": body.roll_deg,
        "pitch_deg": body.pitch_deg,
        "accuracy_m": body.accuracy_m,
    }
    geo = geolocate.locate_detection(
        meta=meta,
        slant_range_m=body.slant_range_m,
        image_height=body.image_height,
        row=body.row,
    )
    if geo.get("known") and geo.get("lat") is not None and geo.get("lon") is not None:
        ell = geo.get("ellipse") or {}
        if ell:
            poly = geolocate.ellipse_polygon(
                lat=geo["lat"],
                lon=geo["lon"],
                semi_major_m=ell.get("semi_major_m", 5.0),
                semi_minor_m=ell.get("semi_minor_m", 5.0),
                rotation_deg=ell.get("rotation_deg", 0.0),
            )
            geo["ellipse_polygon_geojson"] = {
                "type": "Polygon",
                "coordinates": [poly],
            }
    return geo
