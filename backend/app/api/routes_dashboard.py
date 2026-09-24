from __future__ import annotations

from collections import Counter

from fastapi import APIRouter

from .. import db

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("")
def dashboard():
    surveys = db.list_surveys()
    all_dets = db.get_detections()
    by_class = Counter(d["class_name"] for d in all_dets)
    by_status = Counter(d["status"] for d in all_dets)
    by_tier = Counter(d.get("priority", {}).get("tier", "low") for d in all_dets)
    image_count = sum(len(db.get_images(s["id"])) for s in surveys)
    recent = []
    for s in surveys[:6]:
        job = db.latest_job(s["id"])
        recent.append(
            {
                "id": s["id"],
                "name": s["name"],
                "created_at": s["created_at"],
                "image_count": len(db.get_images(s["id"])),
                "detection_count": sum(1 for d in all_dets if d["survey_id"] == s["id"]),
                "job_status": job["status"] if job else None,
            }
        )
    return {
        "survey_count": len(surveys),
        "image_count": image_count,
        "detection_count": len(all_dets),
        "by_class": dict(by_class),
        "by_status": dict(by_status),
        "by_priority_tier": dict(by_tier),
        "human_review_count": by_status.get("human_review_required", 0),
        "recent_surveys": recent,
    }
