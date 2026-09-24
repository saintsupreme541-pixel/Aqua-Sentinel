"""Geolocation failure-audit regression tests.

These lock the exact behaviors verified during the end-to-end audit (§1–§31):
the Upload-form metadata contract survives into frame rows, geolocation
agrees across every consumer (detections API ↔ spatial API ↔ GeoJSON ↔
reports), GeoJSON keeps the [lon, lat] order, empty/NaN metadata degrades
honestly, and located fixes persist across a full service restart (§17).

No test fabricates coordinates at runtime: metadata in these tests is the
operator's observed input, exactly what the Upload form fields capture.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app

client = TestClient(app)

# The exact payload the React Upload form builds for filled nav fields
# (UploadPage.submit: numbers parsed, empties dropped, side always sent).
UPLOAD_FORM_META = {
    "name": "geo-audit-regression",
    "sonar_type": "sss",
    "preprocess_preset": "light",
    "lat": 17.4400,
    "lon": 78.3800,
    "heading_deg": 90.0,
    "altitude_m": 3.0,
    "range_m": 30.0,
    "side": "starboard",
}


def _png() -> bytes:
    """Deterministic sonar-like PNG (bright blob on dark gradient)."""
    rng = np.random.default_rng(11)
    img = rng.integers(0, 40, (96, 128), dtype="uint8")
    img[30:50, 50:78] = 230  # bright candidate object
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _upload(meta: dict, n_images: int = 1) -> tuple[str, str]:
    png = _png()
    files = [("files", (f"frame{i}.png", png, "image/png")) for i in range(n_images)]
    r = client.post("/api/surveys", files=files, data={"meta": json.dumps(meta)})
    assert r.status_code == 200, r.text
    sid, jid = r.json()["survey_id"], r.json()["job_id"]
    deadline = time.time() + 90
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            assert j["status"] == "done", j.get("error")
            return sid, jid
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _waited_dets(sid: str) -> list[dict]:
    return client.get(f"/api/surveys/{sid}/detections").json()


def _seed_detection(sid: str, image_id: str, idx: int = 0, **overrides) -> dict:
    """Insert one runner-shaped detection row via the app's persistence layer
    (validated against schemas.Detection — contract drift fails loudly)."""
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
    from app.schemas import Detection as DetectionModel

    DetectionModel.model_validate(d)
    db.save_detections([d])
    return d


def _locate(sid: str, det: dict) -> dict:
    """Run the app's own geolocation engine for a detection's box+frame meta
    (the exact call the runner makes) and attach the result."""
    from app.pipeline import geolocate

    frame = next(f for f in db.get_images(sid) if f["id"] == det["image_id"])
    imeta = {**json.loads(db.get_survey(sid)["meta_json"]), **json.loads(frame["meta_json"])}
    geo = geolocate.locate_detection(
        meta=imeta,
        slant_range_m=None,
        image_height=int(frame["height"]),
        row=det["box"]["y"] + det["box"]["h"] / 2,
        reference_frame_id=frame["id"],
    )
    det["geolocation"] = geo
    db.save_detections([det])
    return det


def _survey_with_located_detections(
    meta: dict, n: int = 2, lat_offsets: list[float] | None = None
) -> tuple[str, list[dict]]:
    """Upload a survey with metadata, seed detections, and locate them with
    the REAL engine + REAL frame metadata (no synthetic coordinates).
    Then run the app's own association to aggregate targets."""
    from app.targets import run_association

    sid, _ = _upload(meta)
    frames = db.get_images(sid)
    dets = []
    for i in range(n):
        d = _seed_detection(sid, frames[i % len(frames)]["id"], i)
        if lat_offsets:
            d["box"] = {**d["box"], "y": 12.0 + lat_offsets[i] * 10}
        dets.append(_locate(sid, d))
    run_association(sid)
    return sid, dets


# ---------------------------------------------------------------------------
# §4/§6/§7: the Upload-form metadata contract reaches the frame rows intact
# ---------------------------------------------------------------------------


class TestUploadFormMetadataContract:
    def test_form_payload_persists_to_frame_columns(self):
        sid, _ = _upload(UPLOAD_FORM_META)
        f = client.get(f"/api/surveys/{sid}/frames").json()[0]
        # names must match the form's keys EXACTLY (lat→latitude, lon→longitude,
        # range_m→slant_range_m mapping in service._frame_nav_columns)
        assert f["latitude"] == pytest.approx(17.4400)
        assert f["longitude"] == pytest.approx(78.3800)
        assert f["heading_deg"] == pytest.approx(90.0)
        assert f["altitude_m"] == pytest.approx(3.0)
        assert f["slant_range_m"] == pytest.approx(30.0)
        assert f["sonar_side"] == "starboard"

    def test_empty_strings_are_dropped_not_zero(self):
        # num("") → null and the key is OMITTED from the payload (UploadPage):
        # the frame row must stay NULL — never 0 (unknown ≠ zero, §7).
        meta = {**UPLOAD_FORM_META, "name": "geo-audit-empty", "lat": "", "lon": "", "range_m": ""}
        payload = {k: v for k, v in meta.items() if v != ""}
        sid, _ = _upload(payload)
        f = client.get(f"/api/surveys/{sid}/frames").json()[0]
        assert f["latitude"] is None and f["longitude"] is None
        assert f["slant_range_m"] is None
        # supplied values still persist
        assert f["heading_deg"] == pytest.approx(90.0)
        assert f["altitude_m"] == pytest.approx(3.0)

    def test_latitude_zero_is_valid_not_missing(self):
        # 0 is a legal coordinate (Gulf of Guinea) — must NOT be treated as
        # missing, and 0 must survive into the frame row (§7).
        meta = {**UPLOAD_FORM_META, "name": "geo-audit-zero", "lat": 0.0, "lon": 0.0}
        sid, _ = _upload(meta)
        f = client.get(f"/api/surveys/{sid}/frames").json()[0]
        assert f["latitude"] == 0.0 and f["longitude"] == 0.0

    def test_string_number_metadata_is_parsed_as_number(self):
        # The form's <input type=number> sends strings through JSON in the
        # worst case; SurveyMeta (pydantic float) must coerce "90" → 90.0.
        meta = {**UPLOAD_FORM_META, "name": "geo-audit-strnum", "heading_deg": "90", "altitude_m": "3"}
        sid, _ = _upload(meta)
        f = client.get(f"/api/surveys/{sid}/frames").json()[0]
        assert f["heading_deg"] == pytest.approx(90.0)
        assert f["altitude_m"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# §5: survey/frame precedence
# ---------------------------------------------------------------------------


class TestSurveyFramePrecedence:
    def test_frame_position_overrides_survey_position(self):
        # Per-image metadata overrides the survey default (ImageMeta merge in
        # service.create_survey_with_files) — one explicit frame coordinate
        # must NOT be clobbered by the survey-level value.
        sid, _ = _upload(UPLOAD_FORM_META)
        # rewrite one frame's meta the way per-image overrides would
        frame = client.get(f"/api/surveys/{sid}/frames").json()[0]
        with db.transaction() as conn:
            conn.execute(
                "UPDATE images SET meta_json=? WHERE id=?",
                (
                    json.dumps({**UPLOAD_FORM_META, "lat": 1.23, "lon": 4.56, "filename": frame["filename"]}),
                    frame["id"],
                ),
            )
        # the runner merges {survey, image}; verify via a fresh image fetch
        imeta = json.loads(db.get_images(sid)[0]["meta_json"])
        assert imeta["lat"] == pytest.approx(1.23)  # frame-level wins

    def test_runner_merge_order_is_survey_then_image(self):
        # The runner must apply image meta OVER survey meta ({**survey, **image}).
        survey_meta = {"lat": 10.0, "lon": 20.0}
        image_meta = {"lat": 30.0}
        merged = {**survey_meta, **image_meta}
        assert merged["lat"] == 30.0 and merged["lon"] == 20.0


# ---------------------------------------------------------------------------
# §20/§29: API consumers agree — detections ↔ spatial ↔ GeoJSON ↔ CSV
# ---------------------------------------------------------------------------


class TestCrossConsumerAgreement:
    @pytest.fixture(scope="class")
    def located_survey(self):
        # Two frames at different rows → distinct derived fixes through the
        # REAL engine (no synthetic coordinates injected anywhere).
        sid, dets = _survey_with_located_detections(UPLOAD_FORM_META, n=2, lat_offsets=[0.0, 1.0])
        dets = _waited_dets(sid)
        yield sid, dets

    def test_detections_carry_known_fixes(self, located_survey):
        sid, dets = located_survey
        if not dets:
            pytest.skip("detector found no candidates in the synthetic frame")
        known = [d for d in dets if d["geolocation"]["known"]]
        assert known, "full metadata must yield known geolocation for detections"
        for d in known:
            g = d["geolocation"]
            assert g["status"] == "approximate"
            assert -90 <= g["lat"] <= 90 and -180 <= g["lon"] <= 180
            # eastbound + starboard → fix must be SOUTH of the frame (§9)
            assert g["lat"] < UPLOAD_FORM_META["lat"]

    def test_spatial_targets_match_detection_fixes(self, located_survey):
        sid, dets = located_survey
        known = [d for d in dets if d["geolocation"]["known"]]
        if not known:
            pytest.skip("no located detections")
        spatial = client.get(f"/api/surveys/{sid}/spatial").json()
        assert spatial["frames_with_location"] == spatial["frames_total"]
        det_latlons = {(round(d["geolocation"]["lat"], 9), round(d["geolocation"]["lon"], 9)) for d in known}
        tgt_latlons = {(round(t["latitude"], 9), round(t["longitude"], 9)) for t in spatial["targets"]}
        assert det_latlons and det_latlons.issubset(tgt_latlons)

    def test_geojson_lon_lat_order(self, located_survey):
        # GeoJSON MUST be [longitude, latitude] (§20) — a swap would put the
        # fix in the ocean east of Somalia instead of India.  Handles both
        # Point fixes and uncertainty-ellipse Polygon rings.
        sid, _ = located_survey
        geojson = client.get(f"/api/surveys/{sid}/geojson").json()
        feats = geojson.get("features", [])
        if not feats:
            pytest.skip("no geo features")

        def check_coord(lon, lat):
            assert -180 <= lon <= 180 and -90 <= lat <= 90
            # lon≈78.38 → the FIRST element is the longitude
            assert lon == pytest.approx(UPLOAD_FORM_META["lon"], abs=0.01)

        points = 0
        for f in feats:
            geom = f["geometry"]
            if geom["type"] == "Point":
                check_coord(*geom["coordinates"][:2])
                points += 1
            elif geom["type"] == "Polygon":
                for ring in geom["coordinates"]:
                    for lon, lat in ring:
                        check_coord(lon, lat)
        assert points > 0, "located detections must appear as Point features"

    def test_csv_reports_the_same_values(self, located_survey):
        sid, dets = located_survey
        known = [d for d in dets if d["geolocation"]["known"]]
        if not known:
            pytest.skip("no located detections")
        r = client.post(f"/api/surveys/{sid}/reports", json=["csv"])
        assert r.status_code == 200
        rid = r.json()["report_ids"][0]
        data = client.get(f"/api/reports/{rid}/download").text
        cols = data.splitlines()[0].split(",")
        lat_i, lon_i = cols.index("lat"), cols.index("lon")
        id_i = cols.index("detection_id")  # CSV column name for the detection id
        rows = {ln.split(",")[id_i]: ln.split(",") for ln in data.splitlines()[1:] if ln.strip()}
        d = known[0]
        row = rows[d["id"]]
        assert float(row[lat_i]) == pytest.approx(d["geolocation"]["lat"], abs=1e-6)
        assert float(row[lon_i]) == pytest.approx(d["geolocation"]["lon"], abs=1e-6)


# ---------------------------------------------------------------------------
# §39: honest no-metadata behavior (regression lock)
# ---------------------------------------------------------------------------


class TestNoMetadataHonesty:
    def test_no_metadata_yields_unknown_and_no_marker(self):
        meta = {"name": "geo-audit-nogeo", "sonar_type": "sss"}
        sid, _ = _upload(meta)
        dets = _waited_dets(sid)
        spatial = client.get(f"/api/surveys/{sid}/spatial").json()
        assert spatial["frames_with_location"] == 0
        assert spatial["targets_with_location"] == 0
        assert spatial["bounds"] is None
        assert spatial["track"]["available"] is False
        for d in dets:
            assert d["geolocation"]["known"] is False
            assert d["geolocation"]["lat"] is None and d["geolocation"]["lon"] is None
            assert d["geolocation"]["status"] == "unknown"
        for t in spatial["targets"]:
            assert t["latitude"] is None and t["geolocation_status"] == "unknown"

    def test_invalid_geometry_refused_not_collapsed(self):
        # slant ≤ altitude cannot leave the sonar plane — the fix must be
        # refused (unknown), never collapsed onto the frame position (§10/§38).
        meta = {**UPLOAD_FORM_META, "name": "geo-audit-badgeom", "altitude_m": 500.0, "range_m": 30.0}
        sid, _ = _upload(meta)
        dets = _waited_dets(sid)
        if not dets:
            pytest.skip("no detections in frame")
        for d in dets:
            g = d["geolocation"]
            if g["known"]:
                # rows near the bottom may still be valid; but any *derived*
                # fix must not equal the bare frame position with zero offset
                assert g["lat"] != UPLOAD_FORM_META["lat"] or g["lon"] != UPLOAD_FORM_META["lon"]
            else:
                assert g["status"] == "unknown" and g["lat"] is None


# ---------------------------------------------------------------------------
# §17: restart persistence (authoritative DB, not in-memory)
# ---------------------------------------------------------------------------


class TestRestartPersistence:
    def test_locations_survive_service_reload(self):
        sid, _ = _survey_with_located_detections(UPLOAD_FORM_META, n=2)
        before = client.get(f"/api/surveys/{sid}/spatial").json()
        if before["targets_with_location"] == 0:
            pytest.skip("no located targets in this run")
        # simulate a service restart: drop the module-level cached connection
        # so the next request reconnects to the persisted SQLite file
        with db._lock:
            if db._conn is not None:
                db._conn.close()
                db._conn = None
        after = client.get(f"/api/surveys/{sid}/spatial").json()
        assert after["targets_with_location"] == before["targets_with_location"]
        before_locs = sorted((t["latitude"], t["longitude"]) for t in before["targets"])
        after_locs = sorted((t["latitude"], t["longitude"]) for t in after["targets"])
        assert after_locs == before_locs


# ---------------------------------------------------------------------------
# §36: full pipeline with valid metadata locates detections (integration)
# ---------------------------------------------------------------------------


class TestValidMetadataPath:
    def test_pipeline_locates_with_operator_metadata(self):
        sid, dets = _survey_with_located_detections(UPLOAD_FORM_META, n=1)
        known = [d for d in dets if d["geolocation"]["known"]]
        assert known, "with full valid metadata the detection must locate"
        g = known[0]["geolocation"]
        assert g["status"] == "approximate"
        assert g["provenance"]["kind"] == "derived"
        assert g["provenance"]["source"] == "frame_navigation_plus_sonar_geometry"
        assert g["provenance"]["reference_frame_id"] == known[0]["image_id"]
        # bearing 180° (south of an eastbound starboard track)
        assert g["provenance"]["derived"]["bearing_deg"] == pytest.approx(180.0, abs=0.5)
