"""Explainable natural-vs-artificial classifier (heuristic baseline).

The heuristic distinguishes manufactured objects from natural seafloor
using *physical* cues that survive across sonar types:

- **intensity homogeneity** — manufactured surfaces return a strong,
  fairly uniform specular echo; rocks/ripples scatter weakly and their
  returns are textured (high coefficient of variation inside the mask);
- **contrast** against the surrounding seabed;
- **contour convexity (solidity)**;
- **contour straightness** — pipes, cylinders and wreck plating produce
  straight contour segments; natural objects rarely do.

The output is ``p_artificial`` in [0, 1].  Scores near 0.5 are ambiguous
and deliberately *do not* drive a confident verdict — the fusion layer
keeps them in “review”.  A trained CNN replaces this module through the
model registry when weights are provided.
"""

from __future__ import annotations

import cv2
import numpy as np


def classify_heuristic(gray: np.ndarray, box: dict, mask: np.ndarray) -> tuple[float, dict]:
    hgt, wid = gray.shape
    x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
    x1, y1 = min(wid, int(box["x"] + box["w"])), min(hgt, int(box["y"] + box["h"]))
    crop = gray[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1] if mask.shape == gray.shape else np.zeros_like(crop, np.uint8)
    if crop.size == 0:
        return 0.5, {"note": "empty crop"}

    mask_px = mask_c > 0
    masked = crop[mask_px]

    # --- 1. intensity homogeneity -------------------------------------------
    cv_val = float(np.std(masked)) / (float(np.mean(masked)) + 1e-6) if masked.size > 20 else 0.4
    homogeneity = min(1.0, max(0.0, 1.0 - cv_val / 0.30))

    # --- 2. contrast vs surrounding seabed -----------------------------------
    ring = np.ones_like(crop, dtype=bool)
    pad = max(4, min(x1 - x0, y1 - y0) // 3)
    ring[max(0, y0 - pad) - y0 : (y1 + pad) - y0, max(0, x0 - pad) - x0 : (x1 + pad) - x0] = False
    ring &= ~mask_px
    bg = crop[ring]
    fg_mean = float(np.mean(masked)) if masked.size else 0.0
    bg_mean = float(np.mean(bg)) if bg.size else 0.0
    contrast = min(1.0, max(0.0, (fg_mean - bg_mean) / 80.0))

    # --- 3. convexity ----------------------------------------------------------
    contours, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity = float(mask_px.sum()) / max(hull_area, 1.0)
    solidity_score = min(1.0, max(0.0, (solidity - 0.35) / 0.45))

    # --- 4. contour straightness ---------------------------------------------
    straightness = _contour_straightness(mask_c)
    straight_score = min(1.0, max(0.0, straightness / 0.5))

    p = 0.38 * homogeneity + 0.22 * contrast + 0.24 * solidity_score + 0.16 * straight_score
    p = min(0.95, max(0.05, p))  # plain Python min/max keeps float typing simple
    features = {
        "homogeneity": round(homogeneity, 3),
        "contrast": round(contrast, 3),
        "solidity": round(solidity, 3),
        "straightness": round(straightness, 3),
        "intensity_cv": round(cv_val, 3),
    }
    return round(p, 3), features


def _contour_straightness(mask: np.ndarray) -> float:
    """Fraction of the contour perimeter lying on straight segments (Hough)."""
    if mask.sum() == 0:
        return 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour_img = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(contour_img, contours, -1, 255, 1)
    perimeter = max(float(np.sum(contour_img > 0)), 1.0)
    lines = cv2.HoughLinesP(contour_img, rho=1, theta=np.pi / 180, threshold=8, minLineLength=6, maxLineGap=2)
    if lines is None:
        return 0.0
    lines = np.asarray(lines)
    if lines.ndim == 3:
        lines = lines[:, 0]
    straight_len = sum(float(np.hypot(x2 - x1, y2 - y1)) for (x1, y1, x2, y2) in lines)
    return float(min(1.0, straight_len / perimeter))
