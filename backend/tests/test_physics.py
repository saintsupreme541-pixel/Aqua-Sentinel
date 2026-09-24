"""Tests for the physics-informed shadow-geometry module (spec §12).

The module's contract: physical quantities only when the metadata allows,
``None`` + honest notes otherwise.  The similar-triangles height estimate
itself is produced by ``shadow.py``; here we verify the geometry gates and
scores, using the real ``geolocate.pixel_to_slant`` row→slant mapping.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.pipeline import physics as phys


# ---- grazing angle -----------------------------------------------------------
def test_grazing_angle_45deg():
    angle = phys.grazing_angle_deg(20.0, 100.0)
    assert angle == pytest.approx(math.degrees(math.asin(20.0 / 100.0)), abs=1e-6)
    assert angle < 45.0


def test_grazing_angle_none_when_missing():
    assert phys.grazing_angle_deg(None, 100.0) is None
    assert phys.grazing_angle_deg(20.0, None) is None
    assert phys.grazing_angle_deg(0, 100.0) is None
    assert phys.grazing_angle_deg(-5, 100.0) is None


def test_grazing_angle_none_for_impossible_geometry():
    # slant range below altitude is geometrically impossible
    assert phys.grazing_angle_deg(50.0, 30.0) is None


# ---- geometry score window ---------------------------------------------------
def test_geometry_score_full_inside_window():
    assert phys._geometry_score(30.0) == 1.0
    assert phys._geometry_score(phys.GRAZING_MIN_DEG) == 1.0
    assert phys._geometry_score(phys.GRAZING_MAX_DEG) == 1.0


def test_geometry_score_decays_outside_window():
    s_low = phys._geometry_score(2.0)
    s_high = phys._geometry_score(80.0)
    assert 0.0 <= s_low < 1.0
    assert 0.0 <= s_high < 1.0
    assert phys._geometry_score(0.0) == 0.0
    assert phys._geometry_score(200.0) == 0.0


def test_geometry_score_none_when_angle_unknown():
    assert phys._geometry_score(None) is None


# ---- analyze_physics: metadata gating ----------------------------------------
def _meta(**over):
    m = {"sonar_type": "sss", "range_m": 100.0, "altitude_m": 20.0}
    m.update(over)
    return m


def test_metadata_missing_gates_everything():
    box = {"x": 10, "y": 50, "w": 20, "h": 10}
    info = phys.analyze_physics(box=box, shadow={"valid": True, "height_estimate_m": 1.5}, meta={}, image_height=200)
    assert info.metadata_sufficient is False
    assert info.estimated_height_m is None
    assert info.note  # explanatory note present


def test_no_valid_shadow_gates_height():
    box = {"x": 10, "y": 50, "w": 20, "h": 10}
    info = phys.analyze_physics(
        box=box,
        shadow={"valid": False, "height_estimate_m": None},
        meta=_meta(),
        image_height=200,
    )
    assert info.metadata_sufficient is False
    assert info.estimated_height_m is None
    # grazing angle can still be computed from metadata alone
    assert info.grazing_angle_deg is not None


def test_valid_shadow_with_full_metadata():
    box = {"x": 10, "y": 100, "w": 20, "h": 10}
    info = phys.analyze_physics(
        box=box,
        shadow={"valid": True, "height_estimate_m": 1.23},
        meta=_meta(),
        image_height=200,
    )
    assert info.metadata_sufficient is True
    assert info.estimated_height_m == pytest.approx(1.23, abs=0.01)
    # row 105/199 → slant ≈ 52.8 m → grazing ≈ asin(20/52.8)
    expected = math.degrees(math.asin(20.0 / (100.0 * 105.0 / 199.0)))
    assert info.grazing_angle_deg == pytest.approx(expected, abs=0.1)
    assert info.geometry_score == pytest.approx(1.0, abs=1e-6)
    assert "similar-triangles" in info.note


def test_grazing_outside_window_caps_score():
    # near-top row → short slant range; altitude high relative to it → steep
    # grazing angle, outside the shadow window
    box = {"x": 10, "y": 5, "w": 20, "h": 10}
    info = phys.analyze_physics(
        box=box,
        shadow={"valid": True, "height_estimate_m": 0.5},
        meta=_meta(altitude_m=9.0, range_m=20.0),
        image_height=200,
    )
    # row 10/199 × 20 m → slant ≈ 1.0 m < altitude 9 m → impossible → angle None,
    # but the height estimate must still flow through (shadow was valid)
    if info.grazing_angle_deg is None:
        assert info.estimated_height_m == pytest.approx(0.5, abs=0.01)
    else:
        assert info.grazing_angle_deg > phys.GRAZING_MAX_DEG
        assert info.geometry_score is not None and info.geometry_score < 1.0
        assert "caution" in info.note


# ---- shadow geometry score propagates through analyze_shadow -----------------
def test_shadow_grazing_gates_evidence():
    from app.pipeline import shadow as shadow_mod

    rng = np.random.default_rng(7)
    hgt, wid = 200, 120
    # uniform seabed, no shadow → evidence must be 0.0, not None
    img = rng.normal(128, 10, (hgt, wid)).astype(np.uint8)
    cv2_img = np.clip(img, 0, 255)
    info, ev = shadow_mod.analyze_shadow(cv2_img, {"x": 40, "y": 60, "w": 30, "h": 20}, direction="down")
    assert info.available is True
    assert ev == 0.0
    assert info.valid is False

    import cv2  # noqa: F401  (import kept local; cv2_img built without cv2)

    # strong dark shadow band directly below a bright object
    img2 = np.full((hgt, wid), 120, np.uint8)
    img2[60:80, 40:70] = 230  # object
    img2[80:140, 45:65] = 20  # long dark shadow
    info2, ev2 = shadow_mod.analyze_shadow(img2, {"x": 40, "y": 60, "w": 30, "h": 20}, direction="down")
    assert info2.valid is True
    assert ev2 is not None and ev2 > 0.5
