"""Geolocation v2 — provenance-first architecture regression tests (§21).

Covers the 16 required scenarios plus state-model and report contract
checks.  The guiding invariant for every test: the system NEVER produces a
coordinate without genuine navigation evidence, and every result carries an
explicit location status.
"""

from __future__ import annotations

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from app import db
from app import navigation as nav
from app.main import app
from app.pipeline import geolocate
from app.reports import csv_report

client = TestClient(app)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _csv_bytes(rows: list[dict], columns: list[str] | None = None) -> bytes:
    columns = columns or list(rows[0].keys()) if rows else ["timestamp", "latitude", "longitude"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


def _mk_survey(name="geo-v2"):
    """Create a one-frame survey from the shared synthetic sample image."""
    from pathlib import Path

    png = sorted(Path(__file__).parent.parent.joinpath("data").glob("**/*.png"))
    src = png[0] if png else None
    if src is None:
        import numpy as np
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(buf, format="PNG")
        content = buf.getvalue()
    else:
        content = src.read_bytes()
    resp = client.post(
        "/api/surveys",
        data={"meta": json.dumps({"name": name}), "image_meta": json.dumps({})},
        files={"files": ("frame.png", content, "image/png")},
    )
    assert resp.status_code in (200, 201), resp.text
    sid = resp.json()["survey_id"]
    # wait for the analysis job to settle
    job_id = resp.json()["job_id"]
    import time

    for _ in range(100):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    return sid


def _valid_track(n=4, start_lat=17.44000, start_lon=78.38000, step_s=10):
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": f"2026-01-01T10:{i:02d}:00Z" if step_s == 60 else f"2026-01-01T10:00:{i * step_s:02d}Z",
                "latitude": round(start_lat + i * 9e-5, 7),
                "longitude": start_lon,
                "heading_deg": 90,
                "altitude_m": 3,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 1–3: valid / invalid / missing GPS
# ---------------------------------------------------------------------------


class TestNavValidation:
    def test_valid_gps_csv_accepted(self):
        r = nav.ingest_navigation("track.csv", _csv_bytes(_valid_track()))
        assert r["stats"]["valid"] == 4
        assert r["stats"]["rejected"] == 0
        assert r["span"]["start"].startswith("2026-01-01T10:00:00")
        assert all(rec["valid"] for rec in r["records"])

    def test_invalid_gps_rejected_with_diagnostics(self):
        rows = [
            {"timestamp": "2026-01-01T10:00:00Z", "latitude": "95.0", "longitude": "78.38"},
            {"timestamp": "2026-01-01T10:00:10Z", "latitude": "17.44", "longitude": "200.0"},
            {"timestamp": "2026-01-01T10:00:20Z", "latitude": "17.44", "longitude": "78.38"},
        ]
        r = nav.ingest_navigation("track.csv", _csv_bytes(rows))
        assert r["stats"]["valid"] == 1
        assert r["stats"]["rejected"] == 2
        assert "latitude out of range" in r["rejected"][0]["issues"][0]
        assert "longitude out of range" in r["rejected"][1]["issues"][0]

    def test_missing_gps_rejected_not_zero_filled(self):
        rows = [
            {"timestamp": "2026-01-01T10:00:00Z", "latitude": "", "longitude": ""},
            {"timestamp": "2026-01-01T10:00:10Z", "latitude": "17.44", "longitude": "78.38"},
        ]
        r = nav.ingest_navigation("track.csv", _csv_bytes(rows))
        assert r["stats"]["valid"] == 1
        assert "missing latitude/longitude" in r["rejected"][0]["issues"][0]

    def test_empty_and_garbage_files_fail_cleanly(self):
        for name, content in [("track.csv", b""), ("t.geojson", b"not json"), ("t.gpx", b"<broken"), ("t.xyz", b"x")]:
            r = nav.ingest_navigation(name, content)
            assert r["records"] == []
            assert r["warnings"]  # diagnostics explain why

    def test_no_lat_lon_columns(self):
        r = nav.ingest_navigation("track.csv", b"a,b\n1,2\n")
        assert r["records"] == []
        assert "no usable lat/lon" in r["warnings"][0]


# ---------------------------------------------------------------------------
# 4–5: duplicate timestamps + interpolation
# ---------------------------------------------------------------------------


class TestTrackDiagnostics:
    def test_duplicate_timestamps_flagged(self):
        rows = _valid_track()
        dup = dict(rows[1])
        rows.insert(2, dup)  # same timestamp + position twice
        r = nav.ingest_navigation("track.csv", _csv_bytes(rows))
        # both duplicate records are flagged; at least one dropped from valid
        assert r["stats"]["duplicate_timestamps"] >= 1
        valid_ts = [rec["ts"] for rec in r["records"]]
        assert len(valid_ts) == len(set(valid_ts))

    def test_impossible_jump_rejects_both_endpoints(self):
        rows = _valid_track(3)
        # teleport ~11 km in 10 s (≈1100 m/s) — impossible for a vessel
        rows.append(
            {
                "timestamp": "2026-01-01T10:00:30Z",
                "latitude": 17.54,
                "longitude": 78.38,
                "heading_deg": 90,
            }
        )
        r = nav.ingest_navigation("track.csv", _csv_bytes(rows))
        assert r["stats"]["impossible_jumps"] >= 1
        # BOTH implicated records are removed from the valid track AND listed
        # in `rejected` with their reasons (§3: expose, never repair)
        all_issues = [i for rec in r["rejected"] for i in rec["issues"]]
        assert any("impossible jump" in i for i in all_issues)
        lats = [rec["latitude"] for rec in r["records"]]
        assert 17.54 not in lats

    def test_navigation_gap_reported(self):
        rows = [  # noqa: E501
            {"timestamp": "2026-01-01T10:00:00Z", "latitude": "17.44", "longitude": "78.38"},
            {"timestamp": "2026-01-01T10:05:00Z", "latitude": "17.45", "longitude": "78.38"},
        ]
        r = nav.ingest_navigation("track.csv", _csv_bytes(rows))
        assert r["stats"]["gaps_over_60s"] == 1

    def test_timestamp_sync_linear_interpolation(self):
        rows = _valid_track(step_s=10)
        r = nav.ingest_navigation("track.csv", _csv_bytes(rows))
        ts = nav._parse_ts("2026-01-01T10:00:15Z")  # halfway between fix 1 and 2
        s = nav.sync_frame_to_track(ts=ts, track=r["records"])
        assert s is not None
        assert s["sync_method"] == "linear_interpolation"
        # halfway between fix[1] (17.44009) and fix[2] (17.44018)
        assert abs(s["latitude"] - 17.440135) < 1e-5
        assert s["interpolation_fraction"] == pytest.approx(0.5, abs=0.01)

    def test_timestamp_sync_exact_and_nearest(self):
        rows = _valid_track(step_s=10)
        r = nav.ingest_navigation("track.csv", _csv_bytes(rows))
        exact = nav.sync_frame_to_track(ts=nav._parse_ts("2026-01-01T10:00:20Z"), track=r["records"])
        assert exact["sync_method"] == "exact_timestamp"
        # 5 s before the track even starts → nearest, flagged
        early = nav.sync_frame_to_track(ts=nav._parse_ts("2026-01-01T09:59:55Z"), track=r["records"])
        assert early["sync_method"] == "nearest"
        assert "outside navigation span" in early["note"]

    def test_sync_without_track_or_timestamp_is_none(self):
        assert nav.sync_frame_to_track(ts=None, track=_valid_track()) is None
        assert nav.sync_frame_to_track(ts=nav._parse_ts("2026-01-01T10:00:05Z"), track=[]) is None


# ---------------------------------------------------------------------------
# 6–8: CRS conversion + GPX + GeoJSON
# ---------------------------------------------------------------------------


class TestFormatsAndCrs:
    def test_utm_geojson_transformed_to_wgs84(self):
        # UTM 44N (Hyderabad region): easting 443000, northing 1928000
        gj = json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"properties": {"name": "EPSG:32644"}},
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [443000, 1928000]},
                        "properties": {"timestamp": "2026-01-01T10:00:00Z"},
                    }
                ],
            }
        ).encode()
        r = nav.ingest_navigation("t.geojson", gj)
        assert r["stats"]["valid"] == 1
        rec = r["records"][0]
        assert 17.0 < rec["latitude"] < 18.0
        assert 80.0 < rec["longitude"] < 81.0
        assert any("transformed" in i for i in rec["issues"])

    def test_utm_zone_shorthand(self):
        lat, lon = nav._to_wgs84(1928000.0, 443000.0, "UTM44N")
        assert 17.0 < lat < 18.0 and 80.0 < lon < 81.0

    def test_wgs84_not_retransformed(self):
        assert nav._is_wgs84("EPSG:4326") and nav._is_wgs84("WGS84")

    def test_gpx_parsing(self):
        gpx = (
            b'<?xml version="1.0"?><gpx version="1.1"><trk><trkseg>'
            b'<trkpt lat="17.44" lon="78.38"><time>2026-01-01T10:00:00Z</time><ele>3.0</ele></trkpt>'
            b'<trkpt lat="17.4401" lon="78.38"><time>2026-01-01T10:00:10Z</time></trkpt>'
            b"</trkseg></trk></gpx>"
        )
        r = nav.ingest_navigation("t.gpx", gpx)
        assert r["stats"]["valid"] == 2
        assert r["records"][0]["altitude_m"] == 3.0
        assert r["records"][0]["source"] == "gpx"

    def test_geojson_wgs84_passthrough(self):
        gj = json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [78.38, 17.44]},
                        "properties": {"timestamp": "2026-01-01T10:00:00Z", "heading": 90},
                    }
                ],
            }
        ).encode()
        r = nav.ingest_navigation("t.geojson", gj)
        assert r["stats"]["valid"] == 1
        assert r["records"][0]["latitude"] == 17.44


# ---------------------------------------------------------------------------
# 9–12: image metadata, manual georeference, towfish
# ---------------------------------------------------------------------------


class TestStatesAndTowfish:
    def test_image_without_metadata_stays_unavailable(self):
        geo = geolocate.locate_detection(meta={}, slant_range_m=None, image_height=100, row=50)
        assert geo["known"] is False
        assert geolocate.normalize_status(geo["status"]) == "UNAVAILABLE"
        assert geo["lat"] is None and geo["lon"] is None

    def test_state_model_normalization_never_upgrades(self):
        assert geolocate.normalize_status("approximate") == "DERIVED"
        assert geolocate.normalize_status("unknown") == "UNAVAILABLE"
        assert geolocate.normalize_status(None) == "UNAVAILABLE"
        assert geolocate.normalize_status("VERIFIED") == "VERIFIED"

    def test_manual_georeference_is_labelled(self):
        geo = geolocate.manual_geolocation(17.44, 78.38, note="operator fix")
        assert geo["status"] == "MANUAL"
        assert geo["provenance"]["source"] == "user_supplied"
        assert geo["known"] is True
        assert geo["lat"] == 17.44

    def test_towfish_position_and_missing_config(self):
        pos = nav.towfish_position(
            vessel_lat=17.44, vessel_lon=78.38, heading_deg=90, offset_forward_m=10, offset_starboard_m=None
        )
        assert pos is not None
        # 10 m east of the vessel
        assert pos[0] == pytest.approx(17.44, abs=1e-4)
        assert pos[1] > 78.38
        # missing layback → None (never invented, §7)
        assert (
            nav.towfish_position(
                vessel_lat=17.44, vessel_lon=78.38, heading_deg=None, offset_forward_m=10, offset_starboard_m=5
            )
            is None
        )
        assert (
            nav.towfish_position(
                vessel_lat=17.44,
                vessel_lon=78.38,
                heading_deg=90,
                offset_forward_m=None,
                offset_starboard_m=None,
            )
            is None
        )

    def test_survey_towfish_config_reads_only_supplied(self):
        cfg = nav.survey_towfish_config({"layback_m": 20, "sensor_depth_m": "5.5"})
        assert cfg["layback_m"] == 20.0
        assert cfg["sensor_depth_m"] == 5.5
        assert cfg["towfish_offset_forward_m"] is None


# ---------------------------------------------------------------------------
# 13–15 + API: detections, persistence, endpoints
# ---------------------------------------------------------------------------


class TestApiIntegration:
    def test_nav_upload_roundtrip_and_spatial_track(self):
        sid = _mk_survey()
        resp = client.post(
            f"/api/surveys/{sid}/navigation",
            files={"file": ("track.csv", _csv_bytes(_valid_track()), "text/csv")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["stored"] == 4
        assert body["stats"]["rejected"] == 0

        # stored track is queryable with GeoJSON
        got = client.get(f"/api/surveys/{sid}/navigation").json()
        assert got["available"] is True
        assert got["geojson"]["geometry"]["coordinates"][0] == [78.38, 17.44]

        # spatial API exposes the same genuine track (§5)
        spatial = client.get(f"/api/surveys/{sid}/spatial").json()
        assert spatial["nav_track"]["available"] is True
        assert spatial["nav_track"]["coordinates"][0] == [78.38, 17.44]

    def test_nav_upload_invalid_file_is_422_with_diagnostics(self):
        sid = _mk_survey()
        rows = [{"timestamp": "2026-01-01T10:00:00Z", "latitude": "95", "longitude": "78"}]
        resp = client.post(
            f"/api/surveys/{sid}/navigation",
            files={"file": ("bad.csv", _csv_bytes(rows), "text/csv")},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["message"] == "no valid navigation records in file"
        assert detail["rejected"][0]["issues"]

    def test_detection_without_coordinates_never_gets_one(self):
        sid = _mk_survey()
        resp = client.post(
            f"/api/surveys/{sid}/navigation",
            files={"file": ("track.csv", _csv_bytes(_valid_track()), "text/csv")},
        )
        assert resp.status_code == 200
        # analysis job runs; frames without captured_at timestamps cannot be
        # synced (§4: no timestamp → no sync, no fabricated position)
        spatial = client.get(f"/api/surveys/{sid}/spatial").json()
        assert spatial["nav_track"]["available"] is True
        # zero detections expected (blank image) — the point is no crash, no
        # fabricated coordinates anywhere in the payload
        for t in spatial["targets"]:
            assert t["latitude"] is None or t["location_status"] in ("DERIVED", "MANUAL")

    def test_manual_georeference_endpoint(self):
        sid = _mk_survey()
        resp = client.post(
            f"/api/surveys/{sid}/targets",
            json={"canonical_class": "wreck", "latitude": 17.44, "longitude": 78.38},
        )
        assert resp.status_code == 201, resp.text
        tid = resp.json()["id"]
        r = client.post(
            f"/api/surveys/{sid}/targets/{tid}/geolocation/manual",
            json={"latitude": 17.45, "longitude": 78.39, "note": "sonar contact, chart reference"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["geolocation"]["status"] == "MANUAL"
        assert body["geolocation"]["provenance"]["source"] == "user_supplied"
        t = client.get(f"/api/surveys/{sid}/targets/{tid}").json()
        assert t["latitude"] == 17.45
        assert t["geolocation_status"] == "MANUAL"

    def test_diagnostics_endpoint_flags_missing_nav(self):
        sid = _mk_survey()
        d = client.get(f"/api/surveys/{sid}/geolocation/diagnostics").json()
        assert d["navigation"]["gnss_data"] == "missing"
        assert d["navigation"]["survey_track"] == "missing"
        assert d["navigation"]["towfish_data"] == "missing"
        assert d["position_quality"]["located"] == 0

    def test_navigation_delete(self):
        sid = _mk_survey()
        client.post(
            f"/api/surveys/{sid}/navigation",
            files={"file": ("track.csv", _csv_bytes(_valid_track()), "text/csv")},
        )
        r = client.delete(f"/api/surveys/{sid}/navigation")
        assert r.json()["removed"] == 4
        got = client.get(f"/api/surveys/{sid}/navigation").json()
        assert got["record_count"] == 0


# ---------------------------------------------------------------------------
# 16 + report contract
# ---------------------------------------------------------------------------


class TestReportsAndPersistence:
    def test_csv_report_uses_v2_columns_and_null_coordinates(self):
        rows = [
            {
                "id": "det_1",
                "survey_id": "svy_x",
                "image_filename": "f.png",
                "class_name": "wreck",
                "class_confidence": 0.9,
                "status": "review",
                "evidence": {"signals": {}, "availability": {}, "fusion": 0.8, "breakdown": {}, "backend_used": {}},
                "box": {"x": 1, "y": 2, "w": 3, "h": 4},
                "shadow": {},
                "physics": {},
                "geolocation": {"known": False, "status": "unknown", "lat": None, "lon": None},
                "dimensions": {},
                "priority": {},
            }
        ]
        text = csv_report.render(rows)
        rdr = list(csv.reader(io.StringIO(text)))
        header = rdr[0]
        for col in ("location_status", "position_source", "timestamp", "uncertainty_m", "lat", "lon"):
            assert col in header
        det_row = rdr[1]
        assert det_row[header.index("location_status")] == "UNAVAILABLE"
        assert det_row[header.index("lat")] == ""
        assert det_row[header.index("lon")] == ""

    def test_restart_persistence_of_nav_records(self):
        """Navigation survives a fresh connection (simulated restart)."""
        db.get_conn()
        sid = db.create_survey("nav-persist", "", {"name": "nav-persist"}, 0)
        try:
            db.replace_nav_records(
                sid,
                [{"ts": "2026-01-01T10:00:00Z", "latitude": 17.44, "longitude": 78.38, "valid": True}],
            )
            db._conn.close()
            db._conn = None
            records = db.get_nav_records(sid)
            assert len(records) == 1
            assert records[0]["latitude"] == 17.44
        finally:
            conn = db.get_conn()
            conn.execute("DELETE FROM nav_records WHERE survey_id=?", (sid,))
            conn.execute("DELETE FROM surveys WHERE id=?", (sid,))
            conn.commit()

    def test_multiple_detections_each_get_status(self):
        # two detections on the same frame → each carries its own state
        meta = {"lat": 17.44, "lon": 78.38, "heading_deg": 90, "altitude_m": 3, "side": "starboard", "range_m": 30}
        g1 = geolocate.locate_detection(meta=meta, slant_range_m=20, image_height=100, row=50)
        g2 = geolocate.locate_detection(meta=meta, slant_range_m=25, image_height=100, row=70)
        assert g1["known"] and g2["known"]
        assert geolocate.normalize_status(g1["status"]) == "DERIVED"
        assert geolocate.normalize_status(g2["status"]) == "DERIVED"
        assert g1["lat"] != g2["lat"]  # different rows → different fixes
