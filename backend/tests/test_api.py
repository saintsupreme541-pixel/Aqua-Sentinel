import json
import time

from fastapi.testclient import TestClient

from app import db
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_samples_listed():
    r = client.get("/api/samples")
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()]
    assert "synthetic_sss" in ids


def test_upload_and_job_lifecycle(sample_png):
    with open(sample_png, "rb") as fh:
        files = [("files", (sample_png.name, fh, "image/png"))]
        meta = json.dumps(
            {
                "name": "API test",
                "sonar_type": "sss",
                "lat": 10.0,
                "lon": 20.0,
                "heading_deg": 90.0,
                "altitude_m": 15.0,
                "range_m": 60.0,
                "side": "starboard",
            }
        )
        r = client.post("/api/surveys", files=files, data={"meta": meta})
    assert r.status_code == 200, r.text
    body = r.json()
    survey_id, job_id = body["survey_id"], body["job_id"]

    for _ in range(100):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert j["status"] == "done", j.get("error")

    detail = client.get(f"/api/surveys/{survey_id}").json()
    assert detail["image_count"] == 1
    assert detail["detection_count"] >= 0

    geojson = client.get(f"/api/surveys/{survey_id}/geojson").json()
    assert geojson["type"] == "FeatureCollection"


def test_sample_import_and_reports(sample_png):
    # import the small synthetic sample (1 image at least runs quickly)
    r = client.post("/api/samples/synthetic_sss/import")
    assert r.status_code == 200, r.text
    body = r.json()
    survey_id = body["survey_id"]
    for _ in range(200):
        j = db.latest_job(survey_id)
        if j and j["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert j["status"] == "done"

    r = client.post(f"/api/surveys/{survey_id}/reports", json=["csv", "json", "geojson", "pdf"])
    assert r.status_code == 200, r.text
    report_ids = r.json()["report_ids"]
    assert len(report_ids) == 4
    for rid in report_ids:
        dl = client.get(f"/api/reports/{rid}/download")
        assert dl.status_code == 200
        assert dl.content


def test_unknown_survey_404():
    assert client.get("/api/surveys/nope").status_code == 404
