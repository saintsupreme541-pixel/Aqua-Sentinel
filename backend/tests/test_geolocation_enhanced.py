"""Unit and API integration tests for enhanced Geolocation capabilities.

Verifies:
- Towfish layback offset correction
- Transducer roll & pitch attitude compensation
- WGS84 geodesic destination calculation
- Inverse-variance weighted multi-observation location fusion
- Interactive Sonar Geolocation calculator API endpoint (/api/geolocation/calculate)
"""

import math
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.pipeline import geolocate

client = TestClient(app)


def test_geodesic_destination():
    lat, lon = 15.0, 73.0
    bearing = 90.0
    dist = 1000.0  # 1 km East
    res_lat, res_lon = geolocate.geodesic_destination(lat, lon, bearing, dist)

    assert abs(res_lat - 15.0) < 0.001
    assert res_lon > 73.0
    # 1 km at 15 deg lat is approx 0.0093 deg lon
    assert abs((res_lon - 73.0) - 0.0093) < 0.001


def test_towfish_layback_correction():
    # Vessel at (15.0, 73.0), heading 0 (North), towfish layback = 50m behind vessel (South)
    meta = {
        "lat": 15.0,
        "lon": 73.0,
        "heading_deg": 0.0,
        "altitude_m": 10.0,
        "side": "starboard",
        "layback_m": 50.0,
    }
    # Slant range 20m -> ground range sqrt(400 - 100) = 17.32m East
    res = geolocate.locate_detection(meta=meta, slant_range_m=20.0)

    assert res["known"] is True
    prov = res["provenance"]
    assert "layback_applied_m" in prov["derived"]
    assert prov["derived"]["layback_applied_m"] == 50.0
    assert any("layback" in a for a in prov["assumptions"])

    # Position should be ~50m South and ~17.32m East of vessel
    lat_diff = (res["lat"] - 15.0) * 111320.0  # meters North
    assert -52.0 < lat_diff < -48.0  # 50m South


def test_roll_pitch_compensation():
    meta = {
        "lat": 15.0,
        "lon": 73.0,
        "heading_deg": 90.0,
        "altitude_m": 20.0,
        "side": "port",
        "roll_deg": 10.0,  # 10 deg roll
        "pitch_deg": 5.0,   # 5 deg pitch
    }
    res = geolocate.locate_detection(meta=meta, slant_range_m=30.0)

    assert res["known"] is True
    prov = res["provenance"]
    assert "roll_deg" in prov["derived"]
    assert "pitch_deg" in prov["derived"]
    assert any("roll compensation" in a for a in prov["assumptions"])
    assert any("pitch compensation" in a for a in prov["assumptions"])


def test_fused_target_location():
    obs = [
        {"lat": 15.0001, "lon": 73.0001, "uncertainty_m": 10.0},
        {"lat": 15.0002, "lon": 73.0002, "uncertainty_m": 5.0},   # higher accuracy (lower uncertainty)
    ]
    fused = geolocate.fused_target_location(obs)

    assert fused["available"] is True
    assert fused["n_observations"] == 2
    assert fused["fusion_method"] == "inverse_variance_weighted_least_squares"
    # Weighted mean should be closer to observation #2 (weight ~4x larger)
    assert abs(fused["lat"] - 15.00018) < 0.00005
    # Fused uncertainty should be less than min(uncertainties) = 5.0m -> 1 / sqrt(1/100 + 1/25) = 4.47m
    assert fused["uncertainty_m"] < 5.0


def test_api_calculate_geolocation():
    payload = {
        "latitude": 15.0,
        "longitude": 73.0,
        "heading_deg": 90.0,
        "altitude_m": 15.0,
        "side": "starboard",
        "slant_range_m": 25.0,
        "layback_m": 30.0,
        "roll_deg": 5.0,
        "accuracy_m": 2.5,
    }
    resp = client.post("/api/geolocation/calculate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["known"] is True
    assert data["status"] == "approximate"
    assert "lat" in data and "lon" in data
    assert "ground_range_m" in data
    assert "ellipse_polygon_geojson" in data
    assert data["ellipse_polygon_geojson"]["type"] == "Polygon"
    assert len(data["ellipse_polygon_geojson"]["coordinates"][0]) == 37
