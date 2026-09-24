"""Evidence calibration and fusion.

Every evidence signal is mapped to a probability in [0, 1].  Signals that
could not be computed (e.g. acoustic shadow when geometry is unknown) are
*excluded*, and the fusion weights are renormalised over the available
signals — absence of evidence is not evidence of absence.

Fusion is a logistic opinion pool::

    logit = b0 + Σ w_i · p_i          p = σ(gain · logit)

with weights from config.  These are initial prototype weights, documented
as tunable (docs/metrics.md); ``fit_fusion`` replaces them with learned
weights when labelled data is available.  Until fitted, calibration is the
identity and fusion uses the documented default weights.
"""

from __future__ import annotations

import math

import numpy as np

# NOTE: anomaly is deliberately NOT an evidence signal.  A rock fits the
# seabed background model well, so rewarding "low anomaly" would confirm
# natural objects.  Anomaly is used only as an override in `decide_status`
# (high anomaly + low class confidence ⇒ human review required).
DEFAULT_WEIGHTS: dict[str, float] = {
    "detection": 0.30,
    "segmentation": 0.15,
    "natural": 0.20,
    "shadow": 0.15,
    "physics": 0.10,
    "consistency": 0.10,
}

STATUS_CONFIRMED = "confirmed"
STATUS_REVIEW = "review"
STATUS_CANDIDATE = "candidate"
STATUS_REVIEW_REQUIRED = "human_review_required"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def calibrate(value: float | None, a: float = 1.0, b: float = 0.0) -> float | None:
    """Platt-style calibration; identity by default (a=1, b=0)."""
    if value is None:
        return None
    return sigmoid(a * (value - 0.5) + b)


def fuse(
    signals: dict[str, float | None],
    availability: dict[str, bool] | None = None,
    *,
    weights: dict[str, float] | None = None,
    gain: float = 3.0,
    intercept: float = -0.5,
) -> tuple[float, dict[str, float]]:
    """Fuse calibrated signals → (confidence in [0,1], per-signal breakdown)."""
    w = dict(weights or DEFAULT_WEIGHTS)
    avail = availability or {k: v is not None for k, v in signals.items()}
    active = {k: wk for k, wk in w.items() if avail.get(k) and signals.get(k) is not None}
    total = sum(active.values())
    if total <= 0:
        return 0.0, {}
    logit = intercept
    breakdown: dict[str, float] = {}
    for k, wk in active.items():
        val = signals.get(k)
        if val is None:  # availability filter above keeps these out, but satisfy the type checker
            continue
        contribution = (wk / total) * val
        logit += contribution
        breakdown[k] = round(contribution, 4)
    p = sigmoid(gain * logit)
    return round(p, 4), breakdown


def decide_status(
    fusion: float, anomaly_score: float | None, class_conf: float, *, confirm: float = 0.65, review: float = 0.45
) -> str:
    """Map fusion confidence to an operational verdict.

    ``anomaly_score`` is the raw anomaly score (0 = ordinary, 1 = strongly
    out-of-distribution) — a high anomaly score with low class confidence
    overrides everything: the object may not fit any known class.
    """
    if anomaly_score is not None and anomaly_score >= 0.8 and class_conf < 0.6:
        return STATUS_REVIEW_REQUIRED
    if fusion >= confirm:
        return STATUS_CONFIRMED
    if fusion >= review:
        return STATUS_REVIEW
    return STATUS_CANDIDATE


def fit_fusion(
    samples: list[tuple[dict[str, float | None], float]],
    *,
    epochs: int = 400,
    lr: float = 0.5,
    seed: int = 0,
) -> tuple[dict[str, float], float, float]:
    """Fit fusion weights + intercept + gain on (signals, label) pairs.

    Labels are in [0, 1] (e.g. 1.0 = confirmed human annotation).  Returns
    (weights, intercept, gain).  Used by scripts/evaluate.py to demonstrate
    calibration-driven fusion on validation folds.  ``seed`` is accepted for
    API compatibility (fit is deterministic).
    """
    del seed
    keys = list(DEFAULT_WEIGHTS.keys())
    X = np.array([[s.get(k, 0.0) or 0.0 for k in keys] for s, _ in samples], dtype=float)
    y = np.array([t for _, t in samples], dtype=float).reshape(-1, 1)
    w = np.zeros((len(keys), 1))
    b0 = np.zeros((1, 1))
    gain = np.ones((1, 1))
    for _ in range(epochs):
        logit = gain * (b0 + X @ w)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        err = p - y
        dw = X.T @ (err * p * (1 - p))
        db = np.sum(err * p * (1 - p), keepdims=True)
        dg = np.sum((b0 + X @ w) * err * p * (1 - p), keepdims=True)
        w -= lr * dw / max(len(samples), 1)
        b0 -= lr * db / max(len(samples), 1)
        gain -= lr * dg / max(len(samples), 1)
    return (
        {k: float(w[i, 0]) for i, k in enumerate(keys)},
        float(b0[0, 0]),
        float(gain[0, 0]),
    )


def expected_calibration_error(confidences: list[float], labels: list[float], n_bins: int = 10) -> float:
    """ECE over (confidence, binary label) pairs — reported in metrics."""
    conf = np.asarray(confidences, dtype=float)
    lab = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf)) * abs(conf[mask].mean() - lab[mask].mean())
    return float(ece)
