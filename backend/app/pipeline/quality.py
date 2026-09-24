"""Sonar image quality assessment.

Produces a 0–100 quality score plus flags and per-metric values, and a
metadata-completeness report (GPS / heading / altitude / range / side).
Low quality does not block analysis — it is reported so results can be
interpreted with appropriate caution.
"""

from __future__ import annotations

import numpy as np

from ..schemas import QualityReport


def _noise_estimate(gray: np.ndarray) -> float:
    lap = np.abs(np.diff(gray.astype(np.float32), axis=1))
    return float(np.mean(lap) + 1e-6)


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=64, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0] / hist.sum()
    return float(-(hist * np.log2(hist)).sum())


def assess(gray: np.ndarray, meta: dict) -> QualityReport:
    hgt, wid = gray.shape
    gray_f = gray.astype(np.float32)
    mean = float(gray_f.mean())
    std = float(gray_f.std())
    p2, p98 = np.percentile(gray, [2, 98])
    contrast = (p98 - p2) / 256.0
    noise = _noise_estimate(gray)
    ent = _entropy(gray)
    dark_frac = float((gray < 24).mean())
    bright_frac = float((gray > 232).mean())

    flags: list[str] = []
    if contrast < 0.08:
        flags.append("low_contrast")
    if noise > 40:
        flags.append("high_noise")
    if dark_frac > 0.9:
        flags.append("all_dark")
    if bright_frac > 0.9:
        flags.append("all_bright")
    if mean < 5 or mean > 250:
        flags.append("degenerate_range")

    completeness = {
        "gps": meta.get("lat") is not None and meta.get("lon") is not None,
        "heading": meta.get("heading_deg") is not None,
        "altitude": meta.get("altitude_m") is not None,
        "range": meta.get("range_m") is not None,
        "side": meta.get("side") in ("port", "starboard"),
    }
    meta_score = 100.0 * sum(completeness.values()) / len(completeness)

    dynamic = 100.0 * min(1.0, contrast / 0.35)
    noise_score = 100.0 * min(1.0, max(0.0, 1.0 - noise / 60.0))
    validity = 0.0 if ("all_dark" in flags or "all_bright" in flags) else 100.0

    score = 0.20 * validity + 0.30 * dynamic + 0.20 * noise_score + 0.30 * meta_score
    score = round(min(100.0, max(0.0, score)), 1)

    return QualityReport(
        score=score,
        flags=flags,
        metrics={
            "mean": round(mean, 2),
            "std": round(std, 2),
            "contrast": round(contrast, 4),
            "noise": round(noise, 3),
            "entropy": round(ent, 3),
            "dark_fraction": round(dark_frac, 4),
            "bright_fraction": round(bright_frac, 4),
        },
        metadata_completeness=completeness,
    )
