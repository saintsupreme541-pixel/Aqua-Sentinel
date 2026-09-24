"""Priority-score tests — approved v3 taxonomy only.

ghost_net / pipe / cylinder / net_like have no supervised training data
(semantic review v3) and are NOT valid priority inputs; an object with a
net-like signature surfaces as an anomaly and is prioritised via
``unknown_anomaly``.
"""

from app.pipeline.priority import TYPE_RISK, compute_priority


def test_wreck_outranks_debris():
    wreck = compute_priority(
        class_name="wreck", fusion_confidence=0.9, width_m=5.0, height_m=1.0, dimensions_estimable=True
    )
    junk = compute_priority(
        class_name="debris", fusion_confidence=0.9, width_m=5.0, height_m=1.0, dimensions_estimable=True
    )
    assert wreck.score > junk.score
    assert wreck.factors.type_risk == TYPE_RISK["wreck"]
    assert junk.factors.type_risk == TYPE_RISK["debris"]


def test_anomaly_class_is_a_valid_priority_input():
    p = compute_priority(
        class_name="unknown_anomaly", fusion_confidence=0.7, width_m=None, height_m=None, dimensions_estimable=False
    )
    assert 0.0 <= p.score <= 100.0
    assert p.factors.type_risk == TYPE_RISK["unknown_anomaly"]


def test_score_bounds():
    for cls in ("wreck", "structure", "tire", "debris", "unknown", "rock", "not_a_class"):
        p = compute_priority(
            class_name=cls, fusion_confidence=0.5, width_m=None, height_m=None, dimensions_estimable=False
        )
        assert 0.0 <= p.score <= 100.0
        assert p.tier in ("critical", "high", "medium", "low")


def test_confidence_raises_score():
    lo = compute_priority(
        class_name="tire", fusion_confidence=0.2, width_m=2.0, height_m=1.0, dimensions_estimable=True
    )
    hi = compute_priority(
        class_name="tire", fusion_confidence=0.9, width_m=2.0, height_m=1.0, dimensions_estimable=True
    )
    assert hi.score > lo.score


def test_tiers():
    # Under the approved v3 taxonomy NO supervised class reaches the critical
    # tier: critical previously required ghost_net's 1.0 entanglement risk,
    # and ghost_net has no verified training data. A confident large wreck
    # lands in *high* — honest, not inflated. When genuine net annotations
    # are trained (post-semantic-review-v4), net classes re-enter the risk
    # tables and critical becomes reachable again.
    assert (
        compute_priority(
            class_name="wreck", fusion_confidence=0.99, width_m=30.0, height_m=5.0, dimensions_estimable=True
        ).tier
        == "high"
    )
    # unknown-size debris at very low confidence: env/type risk keeps it at medium floor
    assert (
        compute_priority(
            class_name="debris", fusion_confidence=0.1, width_m=None, height_m=None, dimensions_estimable=False
        ).tier
        == "medium"
    )
    # a low-risk class at zero confidence with no size info lands in low
    assert (
        compute_priority(
            class_name="structure", fusion_confidence=0.0, width_m=None, height_m=None, dimensions_estimable=False
        ).tier
        == "low"
    )
    # a zero-confidence wreck (neutral size/location) stays below the high tier
    assert (
        compute_priority(
            class_name="wreck", fusion_confidence=0.0, width_m=None, height_m=None, dimensions_estimable=False
        ).tier
        == "medium"
    )


def test_factors_weights_sum_to_one():
    from app.pipeline.priority import WEIGHTS

    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
