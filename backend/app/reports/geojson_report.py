from __future__ import annotations

import json
from typing import Any

from ..pipeline import geolocate
from . import collect_survey_data


def render(data: dict) -> str:
    feats: list[dict[str, Any]] = []
    survey = data["survey"]
    meta = data["survey_meta"]
    track = meta.get("track_points") or []
    if len(track) >= 2:
        feats.append(
            {
                "type": "Feature",
                "properties": {"kind": "survey_track", "survey_id": survey["id"], "survey_name": survey["name"]},
                "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in track]},
            }
        )
    for d in data["detections"]:
        geo = d.get("geolocation", {})
        props = {
            "kind": "detection",
            "detection_id": d.get("id"),
            "image": d.get("image_filename"),
            "class": d.get("class_name"),
            "status": d.get("status"),
            "fusion_confidence": d.get("evidence", {}).get("fusion"),
            "priority_score": d.get("priority", {}).get("score"),
            "priority_tier": d.get("priority", {}).get("tier"),
            "dimensions_m": d.get("dimensions", {}),
            "uncertainty_m": geo.get("uncertainty_m"),
        }
        if not geo.get("known"):
            props["note"] = geo.get("note", "geolocation unavailable")
            feats.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": None,
                }
            )
            continue
        feats.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [geo["lon"], geo["lat"]]},
            }
        )
        ellipse = geo.get("ellipse")
        if ellipse:
            poly = geolocate.ellipse_polygon(
                geo["lat"], geo["lon"], ellipse["semi_major_m"], ellipse["semi_minor_m"], ellipse["rotation_deg"]
            )
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"kind": "uncertainty_ellipse", "detection_id": d.get("id"), **ellipse},
                    "geometry": {"type": "Polygon", "coordinates": [poly]},
                }
            )
    return json.dumps(
        {
            "type": "FeatureCollection",
            "name": f"aqua-sentinel-{survey['name']}",
            "generated_from": "estimated survey metadata + sonar geometry (not survey-grade)",
            "features": feats,
        },
        indent=2,
    )


def render_live(survey_id: str) -> str:
    return render(collect_survey_data(survey_id))
