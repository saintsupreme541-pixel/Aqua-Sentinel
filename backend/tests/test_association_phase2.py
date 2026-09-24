"""Phase 2 tests: multi-frame association → persistent targets.

Covers the 12 required cases plus scoring primitives, idempotency,
isolation, transaction safety and the API surface.  Detection fixtures are
synthetic (clearly labeled) and seeded through the app's own persistence
layer — production/runtime code never fabricates data.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import db
from app import targets as targets_mod
from app.association import (
    DEFAULTS,
    compute_consistency,
    geo_signal,
    rank_candidates,
    score_detection_vs_target,
    spatial_signal,
)
from app.main import app
from tests.test_persistence_phase1 import _any_png, _seed_detection

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures: synthetic multi-frame surveys via the real persistence layer
# ---------------------------------------------------------------------------


def _mk_survey(n_frames: int = 3, with_geo: bool = True) -> tuple[str, list[str]]:
    """Create a survey with n real frames (upload → job done); return ids."""
    png = _any_png()
    files = [("files", (f"frame{i}.png", png, "image/png")) for i in range(n_frames)]
    meta = {"name": "phase2", "sonar_type": "sss"}
    if with_geo:
        meta.update(
            {"lat": 54.0, "lon": 10.0, "heading_deg": 90.0, "altitude_m": 10.0, "range_m": 50.0, "side": "starboard"}
        )
    r = client.post("/api/surveys", files=files, data={"meta": __import__("json").dumps(meta)})
    assert r.status_code == 200, r.text
    sid = r.json()["survey_id"]
    import time

    deadline = time.time() + 60
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        if j["status"] in ("done", "failed"):
            assert j["status"] == "done", j.get("error")
            break
        time.sleep(0.1)
    return sid, [im["id"] for im in db.get_images(sid)]


def _det(
    sid: str,
    fid: str,
    idx: int,
    *,
    cls: str = "debris",
    fusion: float = 0.7,
    lat: float | None = None,
    lon: float | None = None,
    box: dict | None = None,
    uncertainty: float | None = 8.0,
) -> dict:
    geo = {
        "known": lat is not None,
        "lat": lat,
        "lon": lon,
        "uncertainty_m": uncertainty if lat is not None else None,
        "ellipse": None,
        "note": "",
    }
    return _seed_detection(
        sid,
        fid,
        idx,
        class_name=cls,
        box=box or {"x": 50.0, "y": 60.0, "w": 20.0, "h": 16.0},
        geolocation=geo,
        evidence={
            "signals": {
                "detection": 0.7,
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
            "fusion": fusion,
            "breakdown": {},
            "backend_used": {"detection": "onnx", "segmentation": "onnx", "classifier": "onnx"},
        },
    )


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def test_class_signal_same_and_different():
    s, reasons = _ = None, None  # placeholder to keep structure obvious
    from app.association import class_signal

    score, reasons = class_signal("debris", "debris")
    assert score == DEFAULTS.class_same_score and reasons == ["same class"]
    score, reasons = class_signal("debris", "wreck")
    assert score == DEFAULTS.class_diff_score and reasons == []


def test_geo_signal_distance_decay_and_unavailable():
    a = {"known": True, "lat": 54.0, "lon": 10.0}
    near, meta_n, _ = geo_signal(a, {"known": True, "lat": 54.00005, "lon": 10.0})  # ~5.5 m
    assert near is not None and near > 0.9 and meta_n["geographic_distance_m"] < 10
    far, meta_f, _ = geo_signal(a, {"known": True, "lat": 54.0005, "lon": 10.0})  # ~55 m
    assert far == 0.0 and meta_f["geographic_distance_m"] > 40
    un, meta_u, reasons = geo_signal(a, {"known": False, "lat": None, "lon": None})
    assert un is None and "target geolocation unavailable" in reasons


def test_spatial_signal_iou():
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    same, meta, _ = spatial_signal(a, a)
    assert same == 1.0 and meta["box_iou"] == 1.0
    far, meta_f, _ = spatial_signal(a, {"x": 100, "y": 100, "w": 10, "h": 10})
    assert far == 0.0 and meta_f["box_iou"] == 0.0


def test_unavailable_evidence_renormalizes_not_zero():
    """Missing geolocation must NOT count as zero — score stays high (CASE 4)."""
    det = {
        "id": "d",
        "class_name": "debris",
        "geolocation": {"known": False, "lat": None, "lon": None},
        "frame_index": 0,
        "box": {"x": 0, "y": 0, "w": 10, "h": 10},
    }
    tgt = {
        "id": "t1",
        "canonical_class": "debris",
        "latitude": None,
        "longitude": None,
        "representative_box": {"x": 1, "y": 1, "w": 10, "h": 10},
        "last_frame_index": 1,
    }
    ev = score_detection_vs_target(det, tgt)
    assert ev["associated"] is True
    assert "geographic" in ev["unavailable"]
    assert ev["signals"]["class"]["score"] == 1.0


# ---------------------------------------------------------------------------
# The 12 required cases
# ---------------------------------------------------------------------------


def test_case1_same_class_nearby_becomes_one_target():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0, box={"x": 50, "y": 60, "w": 20, "h": 16})
    _det(sid, frames[1], 1, lat=54.00002, lon=10.0, box={"x": 52, "y": 61, "w": 20, "h": 16})
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 1
    links = {d["id"]: d["target_id"] for d in db.get_detections(sid)}
    assert links[dets_of(sid)[0]["id"]] == links[dets_of(sid)[1]["id"]] != None  # noqa: E711


def test_case2_different_class_stays_separate():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, cls="structure", lat=54.00001, lon=10.0)
    _det(sid, frames[1], 1, cls="debris", lat=54.00001, lon=10.0)  # same place, wrong class
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 2
    assert {d["class_name"] for d in dets_of(sid)} == {"structure", "debris"}


def test_case3_same_class_far_apart_stays_separate():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.0, lon=10.0)
    _det(sid, frames[1], 1, lat=54.001, lon=10.0)  # ~111 m away, adjacent frame
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 2


def test_case4_missing_gps_still_associates():
    sid, frames = _mk_survey(with_geo=False)
    # same dimensions + near-identical boxes + same class + adjacent frames
    _det(sid, frames[0], 0, lat=None, lon=None, box={"x": 50, "y": 60, "w": 20, "h": 16})
    _det(sid, frames[1], 1, lat=None, lon=None, box={"x": 51, "y": 60, "w": 20, "h": 16})
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 1


def test_case5_missing_gps_weak_spatial_conservative():
    sid, frames = _mk_survey(with_geo=False)
    _det(sid, frames[0], 0, lat=None, lon=None, box={"x": 0, "y": 0, "w": 10, "h": 10})
    _det(sid, frames[1], 1, lat=None, lon=None, box={"x": 90, "y": 90, "w": 10, "h": 10})
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 2  # no merge on weak evidence


def test_case6_ambiguous_two_targets_not_arbitrarily_merged():
    sid, frames = _mk_survey()
    # T1 (frame 0) and T2 (frame 1) equidistant from the new det (frame 2);
    # T1/T2 are ~44 m apart so they seed two separate targets.
    _det(sid, frames[0], 0, lat=54.00000, lon=10.0, fusion=0.6)
    _det(sid, frames[1], 1, lat=54.00040, lon=10.0, fusion=0.6)
    new = _det(sid, frames[2], 2, lat=54.00020, lon=10.0, fusion=0.7)  # ~22 m from each
    s = targets_mod.run_association(sid)
    # the third detection is within the margin of BOTH seeds → ambiguous →
    # it stays unassociated rather than being merged arbitrarily (§19).
    assert len(s["target_ids"]) == 2
    assert s["ambiguous"] == 1
    ev = (db.get_detection_row(new["id"]).get("raw") or {}).get("association", {})
    assert ev.get("status") == "ambiguous" and ev.get("associated") is False
    assert db.get_detection_row(new["id"]).get("target_id") is None


def test_case7_same_survey_valid():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    _det(sid, frames[1], 1, lat=54.00002, lon=10.0)
    s = targets_mod.run_association(sid)
    assert s["survey_id"] == sid and len(s["target_ids"]) == 1


def test_case8_different_survey_rejected():
    sid_a, frames_a = _mk_survey()
    sid_b, frames_b = _mk_survey()
    _det(sid_a, frames_a[0], 0, lat=54.00001, lon=10.0)
    _det(sid_b, frames_b[0], 0, lat=54.00001, lon=10.0)  # identical observation, other survey
    targets_mod.run_association(sid_a)
    targets_mod.run_association(sid_b)
    # targets never share detections across surveys
    for d in dets_of(sid_b):
        t = db.get_target(d["target_id"])
        assert t["survey_id"] == sid_b


def test_case9_idempotent_repeat_run():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    _det(sid, frames[1], 1, lat=54.00002, lon=10.0)
    s1 = targets_mod.run_association(sid)
    before = {d["id"]: d["target_id"] for d in dets_of(sid)}
    s2 = targets_mod.run_association(sid)
    after = {d["id"]: d["target_id"] for d in dets_of(sid)}
    assert sorted(s1["target_ids"]) == sorted(s2["target_ids"])
    assert before == after
    assert s2["already_associated"] == 2 and s2["targets_created"] == 0


def test_case10_zero_detection_frames_ok():
    sid, frames = _mk_survey(n_frames=3)
    _det(sid, frames[2], 0, lat=54.00001, lon=10.0)  # only the last frame has a detection
    s = targets_mod.run_association(sid)
    assert s["frames_processed"] == 3 and len(s["target_ids"]) == 1


def test_case11_single_detection_target_active_not_confirmed():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 1
    t = db.get_target(s["target_ids"][0])
    assert t["status"] == "active"  # never auto-confirmed
    h = targets_mod.target_history(sid, t["id"])
    assert h["n_observations"] == 1  # honestly single-frame


def test_case12_history_in_frame_order():
    sid, frames = _mk_survey(n_frames=4)
    fusions = [0.61, 0.78, 0.84, 0.71]
    for i, f in enumerate(fusions):
        _det(sid, frames[i], i, lat=54.00001 + i * 1e-7, lon=10.0, fusion=f)
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 1
    h = targets_mod.target_history(sid, s["target_ids"][0])
    idxs = [e["frame_index"] for e in h["history"]]
    assert idxs == sorted(idxs) == [0, 1, 2, 3]
    assert h["n_observations"] == 4
    assert h["first_seen_frame_id"] == frames[0] and h["last_seen_frame_id"] == frames[3]
    # representative = highest fusion (0.84 at frame 2)
    assert h["representative_detection_id"] == h["history"][2]["detection_id"]
    assert h["confidence"] == pytest.approx(0.84)


# ---------------------------------------------------------------------------
# Engine-level behaviours
# ---------------------------------------------------------------------------


def test_rank_candidates_ambiguity_rule():
    det = {
        "id": "d",
        "class_name": "debris",
        "geolocation": {"known": True, "lat": 54.0, "lon": 10.0},
        "frame_index": 2,
        "box": None,
    }
    t1 = {
        "id": "T1",
        "canonical_class": "debris",
        "latitude": 54.00005,
        "longitude": 10.0,
        "representative_box": None,
        "last_frame_index": 0,
    }
    t2 = {
        "id": "T2",
        "canonical_class": "debris",
        "latitude": 54.00005,
        "longitude": 10.0,
        "representative_box": None,
        "last_frame_index": 4,
    }
    res = rank_candidates(det, [t1, t2], same_dimensions=False)
    assert res["status"] == "ambiguous" and res["associated"] is False
    assert "ambiguity margin" in res["reasons"][0]


def test_consistency_canonical_multi_image():
    sid, frames = _mk_survey(n_frames=2)
    d1 = _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    _det(sid, frames[1], 1, lat=54.00002, lon=10.0)
    cons = compute_consistency(dets_of(sid))
    assert cons[d1["id"]]["availability"] is True
    assert cons[d1["id"]]["evidence"] == 0.5  # 1 corroborating view → 0.5 (2+ → 1.0; same scale as before Phase 2)
    assert cons[d1["id"]]["n_views"] == 2


def test_consistency_single_image_unavailable():
    sid, frames = _mk_survey(n_frames=1)
    d1 = _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    cons = compute_consistency(dets_of(sid))
    assert cons[d1["id"]]["availability"] is False and cons[d1["id"]]["evidence"] is None


def test_transaction_rollback_on_error():
    """A failing association run must leave no partial writes (§36)."""
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    n_targets_before = len(db.get_targets(sid))

    original = targets_mod._recompute_target_state_conn

    def boom(*a, **k):
        raise RuntimeError("simulated mid-run failure")

    targets_mod._recompute_target_state_conn = boom
    try:
        with pytest.raises(RuntimeError):
            targets_mod.run_association(sid)
    finally:
        targets_mod._recompute_target_state_conn = original

    assert len(db.get_targets(sid)) == n_targets_before  # rolled back
    assert all(d.get("target_id") is None for d in dets_of(sid))  # no partial links

    # and a clean re-run still works (state not corrupted)
    s = targets_mod.run_association(sid)
    assert len(s["target_ids"]) == 1


def test_reset_rebuilds_without_duplicates():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    _det(sid, frames[1], 1, lat=54.00002, lon=10.0)
    s1 = targets_mod.run_association(sid)
    s2 = targets_mod.run_association(sid, reset=True)
    assert len(s1["target_ids"]) == len(s2["target_ids"]) == 1
    assert s2["target_ids"] != s1["target_ids"]  # rebuilt (new ids)...
    links = {d["target_id"] for d in dets_of(sid)}
    assert links == {s2["target_ids"][0]}  # ...but consistent again
    assert s2["already_associated"] == 0  # nothing was reused after reset


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_api_associate_and_history():
    sid, frames = _mk_survey(n_frames=2)
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    _det(sid, frames[1], 1, lat=54.00002, lon=10.0)

    r = client.post(f"/api/surveys/{sid}/associate", json={})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["frames_processed"] == 2 and s["detections_processed"] == 2
    assert len(s["target_ids"]) == 1
    assert "accept_threshold" in s["config"]

    tid = s["target_ids"][0]
    r2 = client.get(f"/api/surveys/{sid}/targets/{tid}/history")
    assert r2.status_code == 200
    h = r2.json()
    assert h["n_observations"] == 2
    assert [e["frame_index"] for e in h["history"]] == [0, 1]
    assert h["confidence_definition"]  # documented, not a bare number

    # idempotent through the API too
    r3 = client.post(f"/api/surveys/{sid}/associate", json={}).json()
    assert r3["target_ids"] == [tid] and r3["already_associated"] == 2

    # error paths
    assert client.post("/api/surveys/nope/associate", json={}).status_code == 404
    assert client.get(f"/api/surveys/{sid}/targets/tgt_nope/history").status_code == 404


def test_api_associate_reset():
    sid, frames = _mk_survey()
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    client.post(f"/api/surveys/{sid}/associate", json={})
    r = client.post(f"/api/surveys/{sid}/associate", json={"reset": True}).json()
    assert r["targets_created"] == 1
    assert len(client.get(f"/api/surveys/{sid}/targets").json()) == 1


def test_association_evidence_stored_on_detection():
    sid, frames = _mk_survey(n_frames=2)
    _det(sid, frames[0], 0, lat=54.00001, lon=10.0)
    d2 = _det(sid, frames[1], 1, lat=54.00002, lon=10.0)
    targets_mod.run_association(sid)
    ev = (db.get_detection_row(d2["id"]).get("raw") or {}).get("association")
    assert ev and ev["associated"] is True
    assert ev["signals"]["class"]["available"] is True
    assert ev["signals"]["geographic"]["geographic_distance_m"] < 10
    assert ev["signals"]["frame_proximity"]["frame_gap"] == 1
    assert any("same class" in r for r in ev["reasons"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dets_of(sid: str) -> list[dict]:
    return db.get_detections(sid)
