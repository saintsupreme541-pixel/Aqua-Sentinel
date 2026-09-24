from __future__ import annotations

import csv
import io

from .. import db
from ..pipeline.geolocate import normalize_status

COLUMNS = [
    "detection_id",
    "survey_id",
    "image",
    "class",
    "class_confidence",
    "status",
    "fusion_confidence",
    "box_x",
    "box_y",
    "box_w",
    "box_h",
    "mask_area_frac",
    "evidence_detection",
    "evidence_segmentation",
    "evidence_natural",
    "evidence_shadow",
    "evidence_physics",
    "evidence_consistency",
    "evidence_anomaly",
    "shadow_valid",
    "height_estimate_m",
    "geo_known",
    "lat",
    "lon",
    "uncertainty_m",
    "location_status",
    "position_source",
    "timestamp",
    "geo_bearing_deg",
    "geo_ground_range_m",
    "width_m",
    "length_m",
    "height_m",
    "priority_score",
    "priority_tier",
]


def render(detections: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    for d in detections:
        ev = d.get("evidence", {})
        signals = ev.get("signals", {})
        shadow = d.get("shadow", {})
        geo = d.get("geolocation", {})
        dims = d.get("dimensions", {})
        prio = d.get("priority", {})
        # §17: location_status is always present (UNAVAILABLE when missing);
        # lat/lon stay null — never 0,0 placeholders.
        location_status = normalize_status(geo.get("status"))
        prov = geo.get("provenance") or {}
        nav_info = prov.get("navigation") or {}
        position_source = (
            "user_supplied" if location_status == "MANUAL" else nav_info.get("nav_source") or prov.get("source")
        )
        image_id = d.get("image_id")
        image_row = db.get_image(image_id) if isinstance(image_id, str) else None
        writer.writerow(
            [
                d.get("id"),
                d.get("survey_id"),
                d.get("image_filename"),
                d.get("class_name"),
                d.get("class_confidence"),
                d.get("status"),
                ev.get("fusion"),
                d.get("box", {}).get("x"),
                d.get("box", {}).get("y"),
                d.get("box", {}).get("w"),
                d.get("box", {}).get("h"),
                d.get("mask_area_frac"),
                signals.get("detection"),
                signals.get("segmentation"),
                signals.get("natural"),
                signals.get("shadow"),
                signals.get("physics"),
                signals.get("consistency"),
                signals.get("anomaly"),
                shadow.get("valid"),
                shadow.get("height_estimate_m"),
                geo.get("known"),
                geo.get("lat"),
                geo.get("lon"),
                geo.get("uncertainty_m"),
                location_status,
                position_source,
                image_row.get("captured_at") if image_row else None,
                geo.get("bearing_deg"),
                geo.get("ground_range_m"),
                dims.get("width_m"),
                dims.get("length_m"),
                dims.get("height_m"),
                prio.get("score"),
                prio.get("tier"),
            ]
        )
    return buf.getvalue()
