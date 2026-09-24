"""Marine Cleanup Priority Score (0–100).

A transparent, weighted combination of six factors, each normalised to
[0, 1] and explained in the UI so a cleanup team can see *why* an object
was prioritised::

    score = 100 · Σ w_i · factor_i

    w_type=0.20  w_size=0.10  w_entanglement=0.25
    w_env=0.20   w_confidence=0.15  w_location=0.10

Risk tables cover the approved v3 taxonomy plus the anomaly pathway.
``net_like``/``ghost_net``/``pipe``/``cylinder`` are deliberately absent —
no supervised training data exists for them (semantic review v3); an
object with that signature is routed through ``unknown_anomaly``.
"""

from __future__ import annotations

from typing import Literal

from ..schemas import PriorityFactors, PriorityResult

WEIGHTS = {
    "type_risk": 0.20,
    "size": 0.10,
    "entanglement": 0.25,
    "environmental": 0.20,
    "confidence": 0.15,
    "location": 0.10,
}

# Risk weights per supervised class (semantic review v3).
# unknown_anomaly: genuinely unknown ⇒ neutral-ish but flagged for review.
TYPE_RISK = {
    "wreck": 0.85,
    "structure": 0.55,
    "tire": 0.60,
    "debris": 0.60,
    "unknown_anomaly": 0.50,
    "unknown": 0.50,
}

ENTANGLEMENT = {
    "wreck": 0.45,
    "structure": 0.25,
    "tire": 0.30,
    "debris": 0.35,
    "unknown_anomaly": 0.50,
    "unknown": 0.40,
}

ENVIRONMENTAL = {
    "wreck": 0.70,
    "structure": 0.40,
    "tire": 0.45,
    "debris": 0.60,
    "unknown_anomaly": 0.50,
    "unknown": 0.50,
}


def _size_factor(width_m: float | None, height_m: float | None, estimable: bool) -> float:
    if not estimable or width_m is None:
        return 0.5  # neutral when size unknown
    span = max(width_m, height_m or 0.0)
    return min(1.0, span / 20.0)  # 20 m object → full size factor


def compute_priority(
    *,
    class_name: str,
    fusion_confidence: float,
    width_m: float | None,
    height_m: float | None,
    dimensions_estimable: bool,
    location_factor: float = 0.5,
) -> PriorityResult:
    cls = class_name if class_name in TYPE_RISK else "unknown"
    type_risk = TYPE_RISK[cls]
    entanglement = ENTANGLEMENT[cls]
    environmental = ENVIRONMENTAL[cls]
    size = _size_factor(width_m, height_m, dimensions_estimable)
    confidence = float(fusion_confidence)

    factors = PriorityFactors(
        type_risk=round(type_risk, 3),
        size=round(size, 3),
        entanglement=round(entanglement, 3),
        environmental=round(environmental, 3),
        confidence=round(confidence, 3),
        location=round(location_factor, 3),
    )
    score = 100.0 * sum(WEIGHTS[k] * getattr(factors, k) for k in WEIGHTS)
    score = round(min(100.0, max(0.0, score)), 1)
    if score >= 80:
        tier: Literal["critical", "high", "medium", "low"] = "critical"
    elif score >= 60:
        tier = "high"
    elif score >= 40:
        tier = "medium"
    else:
        tier = "low"
    return PriorityResult(score=score, tier=tier, factors=factors)
