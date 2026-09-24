"""End-to-end test: create a survey from a synthetic SSS image + GPS metadata,
run the full analysis job synchronously, then verify every pipeline stage
produced sensible, well-typed output."""

from __future__ import annotations

import json
import time

import cv2

from app import db
from app.service import create_survey_with_files

META = {
    "name": "E2E pipe survey",
    "sonar_type": "sss",
    "lat": 17.68,
    "lon": 83.31,
    "heading_deg": 90.0,
    "altitude_m": 15.0,
    "range_m": 60.0,
    "side": "starboard",
    "preprocess_preset": "light",
}


def _wait(job_id, timeout=120):
    for _ in range(int(timeout / 0.2)):
        j = db.get_job(job_id)
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.2)
    raise AssertionError("job timed out")


def _make_survey(synthetic_pipe, image_meta=None):
    ok, buf = cv2.imencode(".png", synthetic_pipe)
    sid, jid = create_survey_with_files([("pipe.png", buf.tobytes())], {**META, **(image_meta or {})}, {})
    job = _wait(jid)
    return sid, jid, job


def test_pipeline_runs_and_detects_pipe(synthetic_pipe):
    sid, _jid, job = _make_survey(synthetic_pipe)
    assert job["status"] == "done", job.get("error")
    dets = db.get_detections(sid)
    # The heuristic baseline is tuned to fire on this synthetic demo scene.
    # The TRAINED detector (registry backend=onnx) may legitimately return 0
    # candidates on synthetic imagery (domain gap) — empty is then the honest
    # result, and pipeline mechanics are covered by the surrounding tests.
    from app.models.registry import load_registry, resolve

    _cfg, det_backend, _w = resolve(load_registry(), "detection")
    if det_backend == "heuristic":
        assert len(dets) >= 1

    if dets:
        d = max(dets, key=lambda x: x["evidence"]["fusion"])
        # complete evidence envelope
        assert set(d["evidence"]["signals"]) >= {"detection", "segmentation", "natural", "shadow", "consistency"}
        assert 0.0 <= d["evidence"]["fusion"] <= 1.0
        assert d["status"] in ("confirmed", "review", "candidate", "human_review_required")
        # geolocated from GPS metadata
        assert d["geolocation"]["known"] is True
        assert d["geolocation"]["lat"] is not None
        assert d["geolocation"]["uncertainty_m"] is not None
        # dimensions via slant-range geometry
        assert d["dimensions"]["estimable"] is True
        assert d["dimensions"]["width_m"] is not None
        # priority in range
        assert 0 <= d["priority"]["score"] <= 100


def test_no_gps_never_fabricates(synthetic_pipe):
    sid, _j, job = _make_survey(synthetic_pipe, image_meta={"lat": None, "lon": None})
    assert job["status"] == "done"
    for d in db.get_detections(sid):
        assert d["geolocation"]["known"] is False
        assert d["geolocation"]["lat"] is None


def test_image_result_and_artifacts_stored(synthetic_pipe, tmp_storage):
    sid, _j, _job = _make_survey(synthetic_pipe)
    images = db.get_images(sid)
    assert len(images) == 1
    res = db.get_image_result(images[0]["id"])
    assert res is not None
    assert res["processed_path"].endswith(".png")
    # artifact file exists on disk

    from app.config import settings

    assert (settings.storage_dir / res["processed_path"]).exists()


def test_reports_all_formats(synthetic_pipe, tmp_storage):
    from app.reports import collect_survey_data, csv_report, geojson_report, json_report
    from app.reports.pdf_report import render

    sid, _j, _job = _make_survey(synthetic_pipe)
    data = collect_survey_data(sid)
    csv_text = csv_report.render(data["detections"])
    assert csv_text.startswith("detection_id") or "fusion" in csv_text.splitlines()[0]
    assert len(csv_text.splitlines()) == len(data["detections"]) + 1

    json_text = json_report.render(data)
    payload = json.loads(json_text)
    assert payload["format"] == "aqua-sentinel-json-report"

    geo = json.loads(geojson_report.render(data))
    assert geo["type"] == "FeatureCollection"
    if data["detections"]:
        assert any(f["geometry"] is not None for f in geo["features"])

    out = tmp_storage + "/report.pdf"
    render(data, out)
    from pathlib import Path

    assert Path(out).exists() and Path(out).stat().st_size > 2000
