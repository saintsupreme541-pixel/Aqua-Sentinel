import math

import numpy as np

from app.pipeline import geolocate


def test_slant_to_ground_basic():
    # 3-4-5 triangle: slant 5, altitude 3 → ground 4
    assert geolocate.slant_to_ground(5.0, 3.0) == pytest_approx(4.0)


def test_slant_to_ground_vectorized():
    out = geolocate.slant_to_ground(np.array([5.0, 10.0]), np.array([3.0, 6.0]))
    assert out[0] == pytest_approx(4.0)
    assert out[1] == pytest_approx(8.0)


def test_slant_less_than_altitude_clips_to_zero():
    assert geolocate.slant_to_ground(2.0, 5.0) == 0.0


def test_destination_east_from_heading():
    # heading 0 (north), starboard (+90) ⇒ due east
    lat, lon = geolocate.haversine_destination(10.0, 20.0, 90.0, 1000.0)
    assert abs(lat - 10.0) < 0.02  # moving along a parallel keeps latitude approx
    dlon = geolocate.haversine_distance(10.0, 20.0, 10.0, lon)
    assert abs(dlon - 1000.0) < 1.0


def test_locate_detection_starboard_bearing():
    res = geolocate.locate_detection(
        meta={"lat": 10.0, "lon": 20.0, "heading_deg": 0.0, "altitude_m": 10.0, "side": "starboard"},
        slant_range_m=100.0,
    )
    # altitude 10, slant 100 → ground = sqrt(100^2 - 10^2) ≈ 99.5; starboard of heading 0 = east
    assert res["known"]
    assert abs(res["lon"] - 20.0) > 0.0
    assert abs(res["lat"] - 10.0) < 1e-3
    assert res["ground_range_m"] == pytest_approx(math.sqrt(100.0**2 - 10.0**2), rel=1e-3)


def test_zero_altitude_is_not_fabricated():
    # altitude 0 = not recorded (transducer at seabed level is physically impossible)
    res = geolocate.locate_detection(
        meta={"lat": 10.0, "lon": 20.0, "heading_deg": 0.0, "altitude_m": 0.0, "side": "starboard"},
        slant_range_m=100.0,
    )
    assert not res["known"]
    assert "altitude" in res["note"]


def test_missing_gps_never_fabricates():
    res = geolocate.locate_detection(meta={"heading_deg": 90.0, "altitude_m": 10.0, "side": "port"}, slant_range_m=50.0)
    assert not res["known"]
    assert res["lat"] is None and res["lon"] is None
    assert "GPS" in res["note"]


def test_missing_heading_never_fabricates():
    res = geolocate.locate_detection(
        meta={"lat": 10.0, "lon": 20.0, "altitude_m": 10.0, "side": "port"}, slant_range_m=50.0
    )
    assert not res["known"]
    assert "heading" in res["note"]


def test_uncertainty_ellipse_is_sane():
    ell = geolocate._monte_carlo_uncertainty(
        vessel_lat=10.0, vessel_lon=20.0, heading=90.0, side="starboard", slant_range_m=50.0, altitude_m=10.0
    )
    assert 0 < ell["radius_m"] < 50
    e = ell["ellipse"]
    assert e["semi_major_m"] >= e["semi_minor_m"] > 0


def test_ellipse_polygon_geojson_ring():
    pts = geolocate.ellipse_polygon(10.0, 20.0, 100.0, 50.0, 30.0)
    assert len(pts) == 37  # closed ring
    assert pts[0] == pts[-1]
    lons = [p[0] for p in pts]
    assert max(lons) - min(lons) > 1e-6


def pytest_approx(x, rel=None):
    import pytest

    return pytest.approx(x, rel=rel or 1e-6)
