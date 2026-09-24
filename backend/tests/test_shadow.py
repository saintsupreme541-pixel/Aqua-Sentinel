import numpy as np

from app.pipeline import shadow


def _scene_with_shadow(shape=(200, 200)):
    """Bright object at top, dark shadow below it on a mid-grey seabed."""
    img = np.full(shape, 100, np.uint8)
    # object highlight rows 60..100, cols 80..120
    img[60:100, 80:120] = 220
    # acoustic shadow rows 100..140 below the object
    img[100:140, 82:118] = 25
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 3, shape).astype(np.int16), 0, 255).astype(np.uint8)
    return img


def test_valid_shadow_detected():
    img = _scene_with_shadow()
    box = {"x": 80, "y": 60, "w": 40, "h": 40}
    info, ev = shadow.analyze_shadow(img, box, direction="down")
    assert info.available and info.valid
    assert ev is not None and ev > 0.5
    assert info.length_px is not None and info.length_px > 10


def test_no_shadow_is_zero_evidence_not_none():
    img = np.full((200, 200), 100, np.uint8)
    img[60:100, 80:120] = 220  # bright object, flat seabed below (no shadow)
    box = {"x": 80, "y": 60, "w": 40, "h": 40}
    info, ev = shadow.analyze_shadow(img, box, direction="down")
    assert info.available and not info.valid
    assert ev == 0.0  # assessable, but absent


def test_unknown_direction_not_assessable():
    img = _scene_with_shadow()
    box = {"x": 80, "y": 60, "w": 40, "h": 40}
    info, ev = shadow.analyze_shadow(img, box, direction=None)
    assert not info.available and ev is None


def test_height_estimate_geometry():
    """h ≈ H * L / R_far. With a synthetic object + shadow the estimate must be finite & positive."""
    img = _scene_with_shadow()
    box = {"x": 80, "y": 60, "w": 40, "h": 40}
    info, _ = shadow.analyze_shadow(img, box, direction="down", range_m=100.0, altitude_m=10.0, image_height=200)
    assert info.height_estimate_m is not None and 0 < info.height_estimate_m < 10.0


def test_default_direction_by_sonar_type():
    assert shadow.default_direction({"sonar_type": "sss"}) == "down"
    assert shadow.default_direction({"sonar_type": "unknown"}) == "down"
