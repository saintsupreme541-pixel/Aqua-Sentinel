"""Report generation (CSV / JSON / GeoJSON / PDF)."""

from __future__ import annotations

import json
from typing import Any

from .. import db


def collect_survey_data(survey_id: str) -> dict[str, Any]:
    """Assemble everything needed by every report format."""
    survey = db.get_survey(survey_id)
    if not survey:
        raise ValueError(f"unknown survey {survey_id}")
    images = db.get_images(survey_id)
    results = {img["id"]: db.get_image_result(img["id"]) or {} for img in images}
    detections = db.get_detections(survey_id)
    # Phase 3: targets with geolocation provenance (deterministic aggregation
    # from Phase 2/3 recomputation; stored evidence passes through verbatim).
    targets = []
    for t in db.get_targets(survey_id):
        evidence = None
        if t.get("geolocation_evidence_json"):
            try:
                evidence = json.loads(t["geolocation_evidence_json"])
            except ValueError:  # pragma: no cover — defensive
                evidence = None
        targets.append(
            {**{k: v for k, v in t.items() if k != "geolocation_evidence_json"}, "geolocation_evidence": evidence}
        )
    return {
        "survey": {k: v for k, v in survey.items() if k != "meta_json"},
        "survey_meta": json.loads(survey["meta_json"]),
        "images": [
            {
                "id": img["id"],
                "filename": img["filename"],
                "width": img["width"],
                "height": img["height"],
                "meta": json.loads(img["meta_json"]),
                "result": results.get(img["id"]),
            }
            for img in images
        ],
        "detections": detections,
        "targets": targets,
    }
