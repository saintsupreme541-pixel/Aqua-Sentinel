"""Tests for the /api/analyze + /api/detect endpoints (spec §17/§18).

Uses the synthetic-pipe fixture from conftest and asserts the explainable
response contract: per-detection evidence blocks, honest degraded-mode
backends, and no fabricated physics/geolocation.
"""

from __future__ import annotations

import cv2
from fastapi.testclient import TestClient

from app.main import app

META_JSON = (
    '{"sonar_type": "sss", "lat": 17.68, "lon": 83.31, "heading_deg": 90.0, "altitude_m": 15.0, "range_m": 60.0}'
)
META_NOGEO = '{"sonar_type": "sss"}'


def _png_bytes(synthetic_pipe) -> bytes:
    ok, buf = cv2.imencode(".png", synthetic_pipe)
    assert ok
    return buf.tobytes()


def test_analyze_contract_full_metadata(synthetic_pipe, tmp_storage):
    client = TestClient(app)
    r = client.post(
        "/api/analyze",
        files={"file": ("pipe.png", _png_bytes(synthetic_pipe), "image/png")},
        data={"metadata": META_JSON},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["status"] == "completed"
    assert body["analysis_id"].startswith("AQ-")
    assert body["image"]["filename"] == "pipe.png"
    assert body["backends"]["detection"] in ("heuristic", "onnx", "ultralytics")
    assert isinstance(body["backend_notes"], list)
    # degraded mode must be clearly labeled; a fully-trained run may honestly
    # report no notes
    if "heuristic" in body["backends"].values():
        assert body["backend_notes"], "degraded-mode run must state it in backend_notes"
    assert "total" in body["timings_ms"]

    if body["detections"]:
        d = body["detections"][0]
        # explainable per-detection contract (spec §17)
        for key in ("id", "class", "bbox", "yolo_confidence", "segmentation_score", "artificial_probability"):
            assert key in d
        assert set(d["shadow"]) >= {"detected", "length_px", "score", "note"}
        assert set(d["physics"]) >= {"metadata_sufficient", "estimated_height_m", "grazing_angle_deg", "geometry_score"}
        assert "final_confidence" in d and "decision" in d
        assert 0.0 <= d["final_confidence"] <= 1.0
        # taxonomy compliance: only approved v3 classes
        assert d["class"] in {"wreck", "debris", "structure", "tire"}
        # GPS given → geolocation block present (known True/False per geometry gates)
        assert "geolocation" in d


def test_analyze_without_metadata_no_fabrication(synthetic_pipe, tmp_storage):
    client = TestClient(app)
    r = client.post(
        "/api/analyze",
        files={"file": ("pipe.png", _png_bytes(synthetic_pipe), "image/png")},
        data={"metadata": META_NOGEO},
    )
    assert r.status_code == 200
    body = r.json()
    for d in body["detections"]:
        assert d["physics"]["metadata_sufficient"] is False
        assert d["physics"]["estimated_height_m"] is None
        assert d["geolocation"]["known"] is False
        assert d["geolocation"]["lat"] is None


def test_analyze_rejects_bad_metadata(synthetic_pipe, tmp_storage):
    client = TestClient(app)
    r = client.post(
        "/api/analyze",
        files={"file": ("pipe.png", _png_bytes(synthetic_pipe), "image/png")},
        data={"metadata": "{not json"},
    )
    assert r.status_code == 422
    r2 = client.post(
        "/api/analyze",
        files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert r2.status_code == 400


def test_analyze_multi_frame_consistency(synthetic_pipe, tmp_storage):
    client = TestClient(app)
    png = _png_bytes(synthetic_pipe)
    r = client.post(
        "/api/analyze",
        files=[("file", ("f0.png", png, "image/png")), ("frames", ("f1.png", png, "image/png"))],
        data={"metadata": META_JSON},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["images"]) == 2
    for d in body["detections"]:
        assert d["temporal_consistency"] is not None  # same frame twice → high agreement


def test_detect_endpoint_lightweight(synthetic_pipe, tmp_storage):
    client = TestClient(app)
    r = client.post(
        "/api/detect",
        files={"file": ("pipe.png", _png_bytes(synthetic_pipe), "image/png")},
        data={"metadata": META_JSON},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"analysis_id", "backends", "detections"}
    for d in body["detections"]:
        assert set(d) == {"id", "class", "confidence", "bbox"}
        assert d["class"] in {"wreck", "debris", "structure", "tire"}


def test_metadata_endpoint_documents_contract():
    client = TestClient(app)
    r = client.get("/api/metadata")
    assert r.status_code == 200
    body = r.json()
    assert "lat" in body["fields"] and "range_m" in body["fields"]
    assert "never fabricates" in body["policy"]
