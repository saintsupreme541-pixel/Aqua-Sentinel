"""Phase 3 tests: geolocation provenance + target-level spatial intelligence.

Covers spec §34: valid/invalid metadata, missing metadata, deterministic
geographic math (§35 — known synthetic inputs → expected coordinate with
correct hemisphere/sign behavior), provenance structure (§5), statuses
(§4), target aggregation (§7/§8), multi-frame location consistency (§17),
idempotency (§36), survey track (§23), API spatial summary (§22),
cross-survey isolation, and Phase 2 regression.

Geographic math is verified against hand-computed expected values (haversine
destination from a known reference), not just "not null".
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app import db, targets
from app.main import app
from app.pipeline import geolocate

client = TestClient(app)


# ---------------------------------------------------------------------------
# §35: deterministic geographic math — known synthetic inputs
# ---------------------------------------------------------------------------


class TestGeographicMath:
    def test_haversine_destination_bearing_zero_moves_north(self):
        # From the equator, 0° bearing, one degree-worth of arc north → +1° lat.
        deg_arc_m = geolocate.EARTH_RADIUS_M * math.pi / 180.0
        lat, lon = geolocate.haversine_destination(0.0, 0.0, 0.0, deg_arc_m)
        assert lat == pytest.approx(1.0, abs=1e-4)
        assert lon == pytest.approx(0.0, abs=1e-6)

    def test_haversine_destination_bearing_90_moves_east(self):
        # Bearing 90° (east): longitude must INCREASE (correct hemisphere sign).
        deg_arc_m = geolocate.EARTH_RADIUS_M * math.pi / 180.0
        lat, lon = geolocate.haversine_destination(0.0, 0.0, 90.0, deg_arc_m)
        assert lat == pytest.approx(0.0, abs=1e-4)
        assert lon == pytest.approx(1.0, abs=1e-3)
        assert lon > 0.0

    def test_haversine_destination_bearing_270_moves_west(self):
        deg_arc_m = geolocate.EARTH_RADIUS_M * math.pi / 180.0
        lat, lon = geolocate.haversine_destination(0.0, 0.0, 270.0, deg_arc_m)
        assert lat == pytest.approx(0.0, abs=1e-4)
        assert lon == pytest.approx(-1.0, abs=1e-3)
        assert lon < 0.0

    def test_haversine_destination_southern_hemisphere(self):
        # Bearing 180° from the equator crosses into the southern hemisphere.
        deg_arc_m = geolocate.EARTH_RADIUS_M * math.pi / 180.0
        lat, _ = geolocate.haversine_destination(0.0, 0.0, 180.0, deg_arc_m)
        assert lat == pytest.approx(-1.0, abs=1e-4)

    def test_haversine_distance_symmetry(self):
        d1 = geolocate.haversine_distance(17.44, 78.38, 17.45, 78.39)
        d2 = geolocate.haversine_distance(17.45, 78.39, 17.44, 78.38)
        assert d1 == pytest.approx(d2, rel=1e-9)
        assert 1000 < d1 < 2000  # ~1.4 km for 0.01°×0.01° at that latitude

    def test_slant_to_ground_pythagoras(self):
        # 3-4-5 triangle: slant 5 m at altitude 3 m → ground 4 m.
        assert geolocate.slant_to_ground(5.0, 3.0) == pytest.approx(4.0)

    def test_slant_below_altitude_clamps_to_zero(self):
        # Physically impossible (object above the sonar plane) → 0, never NaN.
        assert geolocate.slant_to_ground(2.0, 3.0) == 0.0

    def test_heading_normalization(self):
        assert geolocate.normalize_heading(450.0) == pytest.approx(90.0)
        assert geolocate.normalize_heading(-90.0) == pytest.approx(270.0)
        assert geolocate.normalize_heading(359.9) == pytest.approx(359.9)

    def test_heading_invalid_rejected(self):
        # NaN/inf/non-numeric are rejected — never silently wrapped.
        assert geolocate.normalize_heading(float("nan")) is None
        assert geolocate.normalize_heading(float("inf")) is None
        assert geolocate.normalize_heading("not-a-number") is None
        assert geolocate.normalize_heading(None) is None


# ---------------------------------------------------------------------------
# §15/§34: validation of navigation input
# ---------------------------------------------------------------------------


class TestNavValidation:
    def test_valid_inputs_accepted(self):
        clean, issues = geolocate.validate_nav(
            {"lat": 17.44, "lon": 78.38, "heading_deg": 45, "altitude_m": 3.0, "side": "port", "range_m": 50.0}
        )
        assert not issues
        assert clean["side"] == "port"

    def test_invalid_latitude_rejected(self):
        clean, issues = geolocate.validate_nav(
            {"lat": 95.0, "lon": 0.0, "heading_deg": 0, "altitude_m": 3.0, "side": "port"}
        )
        assert "lat" not in clean
        assert any("latitude" in i for i in issues)

    def test_invalid_longitude_rejected(self):
        clean, issues = geolocate.validate_nav(
            {"lat": 0.0, "lon": -200.0, "heading_deg": 0, "altitude_m": 3.0, "side": "port"}
        )
        assert "lon" not in clean
        assert any("longitude" in i for i in issues)

    def test_missing_gps(self):
        clean, issues = geolocate.validate_nav({"heading_deg": 10, "altitude_m": 3.0, "side": "port"})
        assert "lat" not in clean and "lon" not in clean
        assert any("GPS" in i for i in issues)

    def test_missing_heading(self):
        clean, issues = geolocate.validate_nav({"lat": 0, "lon": 0, "altitude_m": 3.0, "side": "port"})
        assert any("heading" in i for i in issues)

    def test_missing_altitude(self):
        clean, issues = geolocate.validate_nav({"lat": 0, "lon": 0, "heading_deg": 0, "side": "port"})
        assert any("altitude" in i for i in issues)

    def test_missing_sonar_side(self):
        clean, issues = geolocate.validate_nav(
            {"lat": 0, "lon": 0, "heading_deg": 0, "altitude_m": 3.0, "side": "unknown"}
        )
        assert any("side" in i for i in issues)


# ---------------------------------------------------------------------------
# §4/§5: statuses + provenance structure
# ---------------------------------------------------------------------------


class TestGeolocationStatusesAndProvenance:
    def test_complete_metadata_yields_approximate_with_provenance(self):
        meta = {"lat": 17.44, "lon": 78.38, "heading_deg": 0.0, "altitude_m": 3.0, "side": "starboard", "range_m": 50.0}
        geo = geolocate.locate_detection(
            meta=meta, slant_range_m=None, image_height=100, row=50, reference_frame_id="img_x"
        )
        assert geo["known"] is True
        assert geo["status"] == "approximate"
        p = geo["provenance"]
        assert p["kind"] == "derived"
        assert p["source"] == "frame_navigation_plus_sonar_geometry"
        assert p["reference_frame_id"] == "img_x"
        assert p["inputs"]["latitude"] and p["inputs"]["heading_deg"] and p["inputs"]["sonar_side"]
        assert p["derived"]["bearing_deg"] == pytest.approx(90.0, abs=0.1)  # starboard of heading 0
        assert p["derived"]["slant_range_m"] > 0
        assert p["derived"]["ground_range_m"] < p["derived"]["slant_range_m"]  # slant ≠ horizontal
        assert p["assumptions"]

    def test_insufficient_metadata_yields_unknown_with_provenance(self):
        meta = {"lat": 17.44, "lon": 78.38}  # no heading/side/altitude
        geo = geolocate.locate_detection(
            meta=meta, slant_range_m=None, image_height=100, row=50, reference_frame_id="img_x"
        )
        assert geo["known"] is False
        assert geo["status"] == "unknown"
        assert geo["lat"] is None and geo["lon"] is None
        p = geo["provenance"]
        assert p["kind"] == "unavailable"
        assert p["issues"], "provenance must record WHY location is unavailable"

    def test_invalid_latitude_does_not_produce_coordinate(self):
        meta = {"lat": 999.0, "lon": 78.38, "heading_deg": 0.0, "altitude_m": 3.0, "side": "port", "range_m": 50.0}
        geo = geolocate.locate_detection(meta=meta, slant_range_m=None, image_height=100, row=50)
        assert geo["known"] is False
        assert geo["status"] == "unknown"

    def test_invalid_slant_range_rejected(self):
        meta = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "altitude_m": 3.0, "side": "port"}
        geo = geolocate.locate_detection(meta=meta, slant_range_m=-5.0, image_height=100, row=50)
        assert geo["known"] is False and geo["status"] == "unknown"

    def test_slant_below_altitude_refused_not_collapsed(self):
        # Slant 2 m at altitude 3 m cannot leave the sonar plane: the fix must
        # be refused, never silently collapsed onto the frame position.
        meta = {"lat": 17.44, "lon": 78.38, "heading_deg": 90.0, "altitude_m": 3.0, "side": "starboard", "range_m": 5.0}
        geo = geolocate.locate_detection(meta=meta, slant_range_m=2.0, image_height=100, row=50)
        assert geo["known"] is False and geo["status"] == "unknown"
        assert geo["lat"] is None and geo["lon"] is None
        assert any("slant range" in i and "altitude" in i for i in (geo["provenance"]["issues"] or []))

    def test_starboard_vs_port_bearing(self):
        meta = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "altitude_m": 3.0, "side": "starboard", "range_m": 50.0}
        geo_s = geolocate.locate_detection(meta=meta, slant_range_m=5.0, image_height=100, row=50)
        meta["side"] = "port"
        geo_p = geolocate.locate_detection(meta=meta, slant_range_m=5.0, image_height=100, row=50)
        # starboard (heading+90) → east → positive longitude; port → west.
        assert geo_s["lon"] > 0
        assert geo_p["lon"] < 0


# ---------------------------------------------------------------------------
# §17: multi-frame location consistency
# ---------------------------------------------------------------------------


class TestLocationConsistency:
    def test_insufficient_data(self):
        lc = geolocate.location_consistency([(17.0, 78.0)])
        assert lc["available"] is False and lc["status"] == "insufficient_data"
        assert lc["dispersion_m"] is None

    def test_consistent_when_tight(self):
        # ~11 m apart (0.0001° lat) — well inside the 20 m base threshold.
        lc = geolocate.location_consistency([(17.0, 78.0), (17.0001, 78.0)])
        assert lc["available"] is True and lc["status"] == "consistent"

    def test_inconsistent_when_scattered(self):
        # ~1.1 km apart — far beyond the threshold.
        lc = geolocate.location_consistency([(17.0, 78.0), (17.01, 78.0)])
        assert lc["status"] == "inconsistent"

    def test_distinct_from_association(self):
        # §18: consistency is a separate concept — it takes positions, not detections.
        lc = geolocate.location_consistency([], [])
        assert lc["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Target aggregation (§7/§8/§19/§20) — DB-level
# ---------------------------------------------------------------------------


@pytest.fixture()
def survey_with_frames():
    sid = db.create_survey("phase3-geo", "", {"name": "phase3-geo"}, 2)
    f1 = db.add_image(sid, "f0.png", "x/f0.png", {"lat": 17.44, "lon": 78.38}, 100, 100, frame_index=0)
    f2 = db.add_image(sid, "f1.png", "x/f1.png", {"lat": 17.44, "lon": 78.38}, 100, 100, frame_index=1)
    yield sid, f1, f2
    with db.transaction() as conn:
        conn.execute("DELETE FROM surveys WHERE id=?", (sid,))


def _mk_det(sid: str, fid: str, did: str, geo_known: bool, lat: float | None = None, lon: float | None = None) -> dict:
    return {
        "id": did,
        "image_id": fid,
        "survey_id": sid,
        "class_name": "debris",
        "class_confidence": 0.8,
        "box": {"x": 10, "y": 10, "w": 20, "h": 20},
        "evidence": {"signals": {}, "availability": {}, "fusion": 0.7, "breakdown": {}, "backend_used": {}},
        "status": "candidate",
        "geolocation": {
            "known": geo_known,
            "status": "approximate" if geo_known else "unknown",
            "lat": lat,
            "lon": lon,
            "uncertainty_m": 5.0 if geo_known else None,
            "note": "",
        },
        "dimensions": {"estimable": False, "note": ""},
    }


class TestTargetAggregation:
    def test_target_location_from_representative_detection(self, survey_with_frames):
        sid, f1, f2 = survey_with_frames
        db.save_detections(
            [
                _mk_det(sid, f1, "d_a", True, 17.44, 78.38),
                _mk_det(sid, f2, "d_b", True, 17.4401, 78.3801),
            ]
        )
        db.save_detections([]) if False else None
        targets.run_association(sid)
        t = db.get_targets(sid)[0]
        assert t["geolocation_status"] == "approximate"
        assert t["geolocation_source"] == "frame_navigation_plus_sonar_geometry"
        ev = __import__("json").loads(t["geolocation_evidence_json"])
        assert ev["status"] == "approximate"
        assert ev["from_detection_id"] in ("d_a", "d_b")
        assert ev["location_consistency"]["status"] == "consistent"
        assert ev["location_consistency"]["dispersion_m"] < 50.0

    def test_unknown_when_no_located_detections(self, survey_with_frames):
        sid, f1, f2 = survey_with_frames
        db.save_detections([_mk_det(sid, f1, "d_c", False)])
        targets.run_association(sid)
        t = db.get_targets(sid)[0]
        assert t["geolocation_status"] == "unknown"
        assert t["latitude"] is None and t["longitude"] is None

    def test_aggregation_idempotent(self, survey_with_frames):
        sid, f1, f2 = survey_with_frames
        db.save_detections(
            [
                _mk_det(sid, f1, "d_i1", True, 17.44, 78.38),
                _mk_det(sid, f2, "d_i2", True, 17.4401, 78.38),
            ]
        )
        targets.run_association(sid)
        t1 = db.get_targets(sid)[0]
        lat1, lon1, ev1 = t1["latitude"], t1["longitude"], t1["geolocation_evidence_json"]
        targets.run_association(sid)  # second run
        t2 = db.get_targets(sid)[0]
        assert t2["latitude"] == lat1 and t2["longitude"] == lon1
        assert t2["geolocation_evidence_json"] == ev1

    def test_cross_survey_isolation(self):
        s1 = db.create_survey("iso1", "", {"name": "iso1"}, 1)
        s2 = db.create_survey("iso2", "", {"name": "iso2"}, 1)
        f1 = db.add_image(s1, "f.png", "x/f.png", {}, 100, 100, frame_index=0)
        f2 = db.add_image(s2, "f.png", "x/f.png", {}, 100, 100, frame_index=0)
        db.save_detections([_mk_det(s1, f1, "d_s1", True, 17.0, 78.0), _mk_det(s2, f2, "d_s2", True, 17.0, 78.0)])
        targets.run_association(s1)
        targets.run_association(s2)
        t1 = db.get_targets(s1)[0]
        assert "d_s2" not in t1["geolocation_evidence_json"]
        assert db.get_detections(s2)[0]["target_id"] not in (None, t1["id"])
        with db.transaction() as conn:
            conn.execute("DELETE FROM surveys WHERE id IN (?,?)", (s1, s2))

    def test_history_exposes_provenance(self, survey_with_frames):
        sid, f1, f2 = survey_with_frames
        db.save_detections([_mk_det(sid, f1, "d_h", True, 17.44, 78.38)])
        targets.run_association(sid)
        tid = db.get_targets(sid)[0]["id"]
        h = targets.target_history(sid, tid)
        assert h["geolocation_source"] == "frame_navigation_plus_sonar_geometry"
        assert h["geolocation_evidence"] is not None
        assert h["location_consistency"]["status"] == "insufficient_data"  # single located fix
        assert h["history"][0]["geolocation"]["known"] is True


# ---------------------------------------------------------------------------
# §22/§23: survey spatial summary + track
# ---------------------------------------------------------------------------


class TestSpatialAPI:
    def test_track_preserves_frame_order(self):
        sid = db.create_survey("track1", "", {"name": "track1"}, 3)
        for i, (lat, lon) in enumerate([(17.0, 78.0), (17.001, 78.001), (17.002, 78.002)]):
            db.add_image(sid, f"f{i}.png", "x", {}, 10, 10, frame_index=i, latitude=lat, longitude=lon)
        r = client.get(f"/api/surveys/{sid}/spatial")
        assert r.status_code == 200
        body = r.json()
        coords = [p["latitude"] for p in body["track"]["points"]]
        assert len(coords) == 3
        assert coords == sorted(coords)  # frame order preserved
        assert body["track"]["available"] is True
        assert body["frames_with_location"] == 3
        assert body["bounds"] is not None
        with db.transaction() as conn:
            conn.execute("DELETE FROM surveys WHERE id=?", (sid,))

    def test_missing_frame_position_no_fake_track_point(self):
        sid = db.create_survey("track2", "", {"name": "track2"}, 2)
        db.add_image(sid, "f0.png", "x", {}, 10, 10, frame_index=0, latitude=17.0, longitude=78.0)
        db.add_image(sid, "f1.png", "x", {}, 10, 10, frame_index=1)  # no GPS
        r = client.get(f"/api/surveys/{sid}/spatial")
        body = r.json()
        assert body["frames_with_location"] == 1
        assert body["frames_without_location"] == 1
        assert len(body["track"]["points"]) == 1  # gap NOT filled
        assert body["track"]["available"] is False  # single point ≠ track
        with db.transaction() as conn:
            conn.execute("DELETE FROM surveys WHERE id=?", (sid,))

    def test_no_positions_empty_track(self):
        sid = db.create_survey("track3", "", {"name": "track3"}, 1)
        db.add_image(sid, "f0.png", "x", {}, 10, 10, frame_index=0)  # no GPS at all
        body = client.get(f"/api/surveys/{sid}/spatial").json()
        assert body["track"]["coordinates"] == []
        assert body["bounds"] is None

    def test_unknown_survey_404(self):
        assert client.get("/api/surveys/does-not-exist/spatial").status_code == 404


# ---------------------------------------------------------------------------
# Regression: Phase 2 behavior unchanged
# ---------------------------------------------------------------------------


class TestPhase2Unchanged:
    def test_association_run_unchanged_shape(self, survey_with_frames):
        sid, f1, f2 = survey_with_frames
        db.save_detections([_mk_det(sid, f1, "d_r1", False), _mk_det(sid, f2, "d_r2", False)])
        s = targets.run_association(sid)
        assert {"frames_processed", "detections_processed", "targets_created"} <= set(s.keys())
        assert len(db.get_targets(sid)) == s["targets_created"]
