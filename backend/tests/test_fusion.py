import math

import pytest

from app.pipeline import fusion


def test_fusion_monotonic():
    lo = fusion.fuse({"detection": 0.3, "natural": 0.3, "shadow": 0.3}, {})[0]
    hi = fusion.fuse({"detection": 0.9, "natural": 0.9, "shadow": 0.9}, {})[0]
    assert hi > lo


def test_fusion_bounds():
    p, _ = fusion.fuse({"detection": 1.0, "natural": 1.0, "shadow": 1.0, "segmentation": 1.0}, {})
    assert 0.0 <= p <= 1.0


def test_unavailable_signals_renormalized():
    # Without consistency/shadow available, remaining weights renormalize → same total mass
    sig = {"detection": 0.8, "natural": 0.8, "segmentation": 0.8, "shadow": None, "consistency": None}
    avail = {"detection": True, "natural": True, "segmentation": True, "shadow": False, "consistency": False}
    p, breakdown = fusion.fuse(sig, avail)
    assert 0.0 <= p <= 1.0
    assert set(breakdown.keys()) <= {"detection", "natural", "segmentation"}


def test_breakdown_sums_to_weighted_mass():
    sig = {"detection": 1.0, "natural": 1.0}
    avail = {"detection": True, "natural": True}
    p, breakdown = fusion.fuse(sig, avail, gain=1.0, intercept=0.0)
    # logistic with logit = weighted sum; breakdown entries are w_i * s_i
    assert sum(breakdown.values()) == pytest.approx(1.0, abs=1e-6)
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), abs=5e-4)  # fuse rounds to 4 dp


def test_decide_status_override_on_anomaly():
    assert fusion.decide_status(0.9, anomaly_score=0.95, class_conf=0.2) == fusion.STATUS_REVIEW_REQUIRED
    assert fusion.decide_status(0.9, anomaly_score=0.5, class_conf=0.9) == fusion.STATUS_CONFIRMED
    assert fusion.decide_status(0.5, anomaly_score=0.0, class_conf=0.9) == fusion.STATUS_REVIEW
    assert fusion.decide_status(0.3, anomaly_score=0.0, class_conf=0.5) == fusion.STATUS_CANDIDATE


def test_fit_fusion_converges_on_separable_data():
    strong = [
        (
            {"detection": 0.9, "natural": 0.9, "segmentation": 0.9, "shadow": 0.9, "physics": 0.9, "consistency": 0.9},
            1.0,
        )
        for _ in range(40)
    ]
    weak = [
        (
            {"detection": 0.1, "natural": 0.1, "segmentation": 0.1, "shadow": 0.1, "physics": 0.1, "consistency": 0.1},
            0.0,
        )
        for _ in range(40)
    ]
    w, b0, gain = fusion.fit_fusion(strong + weak, epochs=2000, lr=1.0)
    # weights for all signals should be positive; gain positive
    assert gain > 0
    assert all(v > 0 for v in w.values())


def test_ece_zero_for_perfect():
    assert fusion.expected_calibration_error([0.9, 0.7, 0.4], [1, 1, 0]) < 0.6
