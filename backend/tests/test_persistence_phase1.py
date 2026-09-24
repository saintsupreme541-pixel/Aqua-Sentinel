"""Phase 1 persistence tests: Survey → Frame → FrameDetection → PersistentTarget.

Covers the acceptance checklist: hierarchy persistence, target association
(no automatic tracking), reload survivability, API exposure, artifact
references, error paths and cross-survey isolation.

Detection rows for the association tests are seeded through the app's own
persistence functions (validated against ``schemas.Detection``) so the tests
exercise the persistence layer deterministically — not detector behavior.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.main import app
from app.schemas import Detection

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _any_png() -> bytes:
    """One decodable tiny PNG (content irrelevant for hierarchy tests)."""
    img = np.random.default_rng(7).integers(0, 255, (64, 64), dtype="uint8")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        f.seek(0)
        return f.read()


def _upload_survey(meta_extra: dict | None = None, n_images: int = 1) -> str:
    """Create a survey via the real API and wait for its job."""
    png = _any_png()
    files = [("files", (f"frame{i}.png", png, "image/png")) for i in range(n_images)]
    meta = {"name": "phase1 test", "sonar_type": "sss", **(meta_extra or {})}
    r = client.post("/api/surveys", files=files, data={"meta": json.dumps(meta)})
    assert r.status_code == 200, r.text
    sid, job_id = r.json()["survey_id"], r.json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            assert j["status"] == "done", j.get("error")
            return sid
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


def _seed_detection(sid: str, image_id: str, idx: int = 0, **overrides) -> dict:
    """Insert one runner-shaped detection row via the app's persistence layer."""
    det_id = f"det_{image_id.split('_')[-1]}_{idx}"
    d = {
        "id": det_id,
        "image_id": image_id,
        "survey_id": sid,
        "image_filename": f"frame{idx}.png",
        "class_name": "debris",
        "class_confidence": 0.71,
        "box": {"x": 10.0, "y": 12.0, "w": 30.0, "h": 24.0},
        "mask_path": None,
        "mask_area_frac": 0.31,
        "evidence": {
            "signals": {
                "detection": 0.71,
                "segmentation": 1.0,
                "natural": 0.8,
                "shadow": None,
                "physics": None,
                "consistency": None,
            },
            "availability": {
                "detection": True,
                "segmentation": True,
                "natural": True,
                "shadow": False,
                "physics": False,
                "consistency": False,
            },
            "fusion": 0.5,
            "breakdown": {},
            "backend_used": {"detection": "onnx", "segmentation": "onnx", "classifier": "onnx"},
        },
        "status": "candidate",
        "shadow": {
            "available": False,
            "valid": None,
            "length_px": None,
            "length_m": None,
            "height_estimate_m": None,
            "note": "",
        },
        "physics": {
            "metadata_sufficient": False,
            "grazing_angle_deg": None,
            "estimated_height_m": None,
            "geometry_score": None,
            "note": "",
        },
        "geolocation": {
            "known": False,
            "lat": None,
            "lon": None,
            "uncertainty_m": None,
            "ellipse": None,
            "note": "metadata insufficient",
        },
        "dimensions": {"estimable": False, "width_m": None, "length_m": None, "height_m": None, "note": ""},
        "priority": {"score": 10.0, "tier": "low", "factors": {}},
        "raw": {"anomaly_score": 0.5, "classifier_features": {}},
    }
    d.update(overrides)
    Detection.model_validate(d)  # contract drift fails loudly here
    db.save_detections([d])
    return d


def _seeded_survey(n_dets: int = 1) -> tuple[str, str, list[dict]]:
    sid = _upload_survey()
    frame = db.get_images(sid)[0]
    dets = [_seed_detection(sid, frame["id"], i) for i in range(n_dets)]
    return sid, frame["id"], dets


# ---------------------------------------------------------------------------
# 1–4: Survey / Frame / FrameDetection persistence + retrieval
# ---------------------------------------------------------------------------


def test_survey_frame_detection_hierarchy_persisted():
    sid = _upload_survey()
    frames = client.get(f"/api/surveys/{sid}/frames").json()
    assert len(frames) == 1
    f = frames[0]
    assert f["survey_id"] == sid
    assert f["frame_index"] == 0
    assert f["status"] == "done"  # DB records the frame's analysis lifecycle
    assert f["created_at"] is not None

    _seed_detection(sid, f["id"])
    dets = client.get(f"/api/surveys/{sid}/frames/{f['id']}/detections").json()
    survey_dets = client.get(f"/api/surveys/{sid}/detections").json()
    assert len(dets) == len(survey_dets) == 1
    assert dets[0]["id"] == survey_dets[0]["id"]
    assert dets[0]["image_id"] == f["id"]


def test_frame_index_is_ordered():
    sid = _upload_survey(n_images=3)
    frames = client.get(f"/api/surveys/{sid}/frames").json()
    assert [f["frame_index"] for f in frames] == [0, 1, 2]
    assert [f["filename"] for f in frames] == ["frame0.png", "frame1.png", "frame2.png"]


def test_frame_navigation_columns_populated_from_metadata():
    # Values the operator actually supplied must be persisted on the frame.
    sid = _upload_survey(
        {"lat": 54.32, "lon": 10.11, "heading_deg": 45.0, "altitude_m": 12.5, "range_m": 80.0, "side": "port"}
    )
    f = client.get(f"/api/surveys/{sid}/frames").json()[0]
    assert f["latitude"] == pytest.approx(54.32)
    assert f["longitude"] == pytest.approx(10.11)
    assert f["heading_deg"] == pytest.approx(45.0)
    assert f["altitude_m"] == pytest.approx(12.5)
    assert f["slant_range_m"] == pytest.approx(80.0)
    assert f["sonar_side"] == "port"


def test_missing_navigation_stays_null():
    sid = _upload_survey()  # no metadata supplied
    f = client.get(f"/api/surveys/{sid}/frames").json()[0]
    assert f["latitude"] is None
    assert f["longitude"] is None
    assert f["heading_deg"] is None
    assert f["altitude_m"] is None
    assert f["sonar_side"] is None
    assert f["slant_range_m"] is None
    assert f["captured_at"] is None


# ---------------------------------------------------------------------------
# 5–8: Persistent targets + association (no automatic tracking)
# ---------------------------------------------------------------------------


def test_target_create_associate_retrieve():
    sid, frame_id, dets = _seeded_survey()
    det = dets[0]

    # Before the explicit association, no detection carries a target_id.
    assert all(d.get("target_id") is None for d in client.get(f"/api/surveys/{sid}/detections").json())

    r = client.post(
        f"/api/surveys/{sid}/targets",
        json={
            "canonical_class": det["class_name"],
            "representative_detection_id": det["id"],
            "confidence": 0.82,
            "geolocation_status": "unknown",
            "notes": "operator-created target",
        },
    )
    assert r.status_code == 201, r.text
    t = r.json()
    assert t["survey_id"] == sid
    assert t["canonical_class"] == det["class_name"]
    assert t["first_seen_frame_id"] == frame_id
    assert t["last_seen_frame_id"] == frame_id
    assert det["id"] in t["detection_ids"]

    # The detection row now carries the link (reloaded from the DB).
    d = client.get(f"/api/detections/{det['id']}").json()
    assert d["target_id"] == t["id"]

    targets = client.get(f"/api/surveys/{sid}/targets").json()
    assert [x["id"] for x in targets] == [t["id"]]


def test_associate_and_unlink_detection():
    sid, _frame_id, dets = _seeded_survey()
    det = dets[0]
    t = client.post(f"/api/surveys/{sid}/targets", json={"canonical_class": "debris"}).json()

    r = client.post(f"/api/surveys/{sid}/targets/associate", json={"detection_id": det["id"], "target_id": t["id"]})
    assert r.status_code == 200, r.text
    assert client.get(f"/api/detections/{det['id']}").json()["target_id"] == t["id"]

    # Unlink via null target_id
    r = client.post(f"/api/surveys/{sid}/targets/associate", json={"detection_id": det["id"], "target_id": None})
    assert r.status_code == 200
    assert client.get(f"/api/detections/{det['id']}").json()["target_id"] is None


def test_fusion_pass_does_not_clobber_target_link():
    """update_detections must preserve target_id when the fusion pass rewrites rows."""
    sid, _frame_id, dets = _seeded_survey()
    det = dets[0]
    t = client.post(f"/api/surveys/{sid}/targets", json={"canonical_class": "wreck"}).json()
    client.post(f"/api/surveys/{sid}/targets/associate", json={"detection_id": det["id"], "target_id": t["id"]})

    # Simulate the fusion-pass upsert path (no target_id in the dict).
    row = db.get_detection_row(det["id"])
    row.pop("target_id")
    db.update_detections([row])
    assert db.get_detection_row(det["id"])["target_id"] == t["id"]


def test_no_automatic_target_creation():
    """Phase 1: analysis alone must not fabricate targets."""
    sid = _upload_survey()
    assert client.get(f"/api/surveys/{sid}/targets").json() == []


# ---------------------------------------------------------------------------
# 9: Persistence survives a service-layer reload
# ---------------------------------------------------------------------------


def test_persistence_survives_reload_from_disk():
    sid, _frame_id, dets = _seeded_survey()

    # Read straight from the DB file with an independent connection — the same
    # thing another process (or a restarted server) would see.
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id FROM detections WHERE survey_id=?", (sid,)).fetchall()
    surveys = conn.execute("SELECT id FROM surveys WHERE id=?", (sid,)).fetchall()
    frames = conn.execute("SELECT id FROM images WHERE survey_id=?", (sid,)).fetchall()
    conn.close()
    assert {r["id"] for r in rows} == {d["id"] for d in dets}
    assert surveys and frames


# ---------------------------------------------------------------------------
# 10–11: Existing workflows unchanged
# ---------------------------------------------------------------------------


def test_existing_survey_workflow_unchanged(sample_png):
    with open(sample_png, "rb") as fh:
        r = client.post("/api/surveys", files=[("files", (sample_png.name, fh, "image/png"))], data={"meta": "{}"})
    assert r.status_code == 200, r.text
    sid = r.json()["survey_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j.get("error")

    dets = client.get(f"/api/surveys/{sid}/detections").json()
    assert isinstance(dets, list)
    for d in dets:
        assert {"id", "class_name", "evidence", "box"} <= set(d)


def test_existing_reports_still_work(sample_png):
    with open(sample_png, "rb") as fh:
        r = client.post("/api/surveys", files=[("files", ("pipe.png", fh, "image/png"))], data={"meta": "{}"})
    sid = r.json()["survey_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "done"

    r = client.post(f"/api/surveys/{sid}/reports", json=["csv", "json"])
    assert r.status_code == 200, r.text
    for rid in r.json()["report_ids"]:
        dl = client.get(f"/api/reports/{rid}/download")
        assert dl.status_code == 200
        assert dl.content


# ---------------------------------------------------------------------------
# 13–15: Constraints, error paths, cross-survey isolation
# ---------------------------------------------------------------------------


def test_invalid_ids_return_proper_errors():
    r = client.get("/api/surveys/nope/frames")
    assert r.status_code == 404
    r = client.get("/api/surveys/nope/targets")
    assert r.status_code == 404

    sid = _upload_survey()
    assert client.get(f"/api/surveys/{sid}/frames/img_does_not_exist").status_code == 404
    assert client.get(f"/api/surveys/{sid}/frames/{sid}/targets/tgt_does_not_exist").status_code in (404, 405)
    assert client.get(f"/api/surveys/{sid}/targets/tgt_does_not_exist").status_code == 404
    assert client.get("/api/detections/det_nope").status_code == 404


def test_associate_rejects_cross_survey_target():
    sid_a, _f_a, dets_a = _seeded_survey()
    sid_b, _f_b, dets_b = _seeded_survey()
    t_b = client.post(f"/api/surveys/{sid_b}/targets", json={"canonical_class": "debris"}).json()

    # Survey A route trying to use Survey B's target → 404 (scoped lookup)
    r = client.post(
        f"/api/surveys/{sid_a}/targets/associate",
        json={"detection_id": dets_b[0]["id"], "target_id": t_b["id"]},
    )
    assert r.status_code == 404

    # DB-level guard: direct cross-survey assignment is rejected.
    with pytest.raises(ValueError, match="different survey"):
        db.assign_detection_target(dets_a[0]["id"], t_b["id"])

    # Survey A's target list never contains Survey B's target.
    ids_a = {t["id"] for t in client.get(f"/api/surveys/{sid_a}/targets").json()}
    assert t_b["id"] not in ids_a


def test_detection_detail_no_cross_survey_leakage():
    """The old detection_detail scanned every survey's detections; the fixed
    endpoint returns the row's own scope and nothing else."""
    sid_a, _f_a, dets_a = _seeded_survey()
    sid_b, _f_b, dets_b = _seeded_survey()

    d = client.get(f"/api/detections/{dets_b[0]['id']}").json()
    assert d["survey_id"] == sid_b
    assert d["survey_id"] != sid_a
    assert d["image_id"]  # scope present

    d_a = client.get(f"/api/detections/{dets_a[0]['id']}").json()
    assert d_a["survey_id"] == sid_a


def test_cascade_delete_survey_removes_frames_and_detections():
    sid, _frame_id, dets = _seeded_survey()
    assert db.get_detections(sid)
    assert db.get_images(sid)

    conn = sqlite3.connect(str(settings.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DELETE FROM surveys WHERE id=?", (sid,))
    conn.commit()
    conn.close()

    assert db.get_survey(sid) is None
    assert db.get_images(sid) == []
    assert db.get_detections(sid) == []


def test_artifact_references_resolve(sample_png):
    sid = _upload_survey()
    detail = client.get(f"/api/surveys/{sid}").json()
    im = detail["images"][0]
    assert im["raw_url"].startswith("/api/media/")
    r = client.get(im["raw_url"])
    assert r.status_code == 200
    assert r.content
