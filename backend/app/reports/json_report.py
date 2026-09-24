from __future__ import annotations

import json
from collections import Counter


def render(data: dict) -> str:
    payload = {
        "format": "aqua-sentinel-json-report",
        "version": "1.1",
        "survey": {
            "id": data["survey"]["id"],
            "name": data["survey"]["name"],
            "description": data["survey"].get("description", ""),
            "created_at": data["survey"]["created_at"],
            "meta": data["survey_meta"],
        },
        "images": [],
        "detections": data["detections"],
        "targets": data.get("targets", []),
        "geolocation_note": (
            "detection/target coordinates are DERIVED (frame navigation + sonar geometry, "
            "approximate, never survey-grade); frame positions are OBSERVED metadata. "
            "Missing metadata yields status 'unknown' with null coordinates — never fabricated."
        ),
        "summary": _summarize(data),
    }
    for img in data["images"]:
        res = img["result"] or {}
        payload["images"].append(
            {
                "id": img["id"],
                "filename": img["filename"],
                "width": img["width"],
                "height": img["height"],
                "meta": img["meta"],
                "quality": res.get("quality"),
                "preprocess_params": res.get("preprocess_params"),
                "processed_path": res.get("processed_path"),
            }
        )
    return json.dumps(payload, indent=2)


def _summarize(data: dict) -> dict:
    dets = data["detections"]
    return {
        "image_count": len(data["images"]),
        "detection_count": len(dets),
        "by_class": dict(Counter(d["class_name"] for d in dets)),
        "by_status": dict(Counter(d["status"] for d in dets)),
        "by_priority_tier": dict(Counter(d.get("priority", {}).get("tier", "low") for d in dets)),
    }
