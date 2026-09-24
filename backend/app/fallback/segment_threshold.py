"""Threshold-based segmentation fallback.

The threshold is derived from a *ring around the box* (local seabed
statistics), not the box interior — a strong target that dominates its box
would otherwise push mean+k·std above its own values.  Pixels above
bg+k·σ inside the box become the highlight mask; morphological cleanup keeps
the largest component.  ``mask_area_frac`` = mask px / box px feeds the
segmentation-agreement evidence signal.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def segment_heuristic(gray: np.ndarray, box: dict, k: float = 1.6) -> tuple[np.ndarray, float]:
    hgt, wid = gray.shape
    x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
    x1, y1 = min(wid, int(box["x"] + box["w"])), min(hgt, int(box["y"] + box["h"]))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((hgt, wid), np.uint8), 0.0

    # ring background statistics (exclude the box itself)
    pad = max(6, int(0.5 * max(x1 - x0, y1 - y0)))
    bg = np.ones((hgt, wid), dtype=bool)
    bg[y0:y1, x0:x1] = False
    ring_y0, ring_y1 = max(0, y0 - pad), min(hgt, y1 + pad)
    ring_x0, ring_x1 = max(0, x0 - pad), min(wid, x1 + pad)
    bg[ring_y0:ring_y1, ring_x0:ring_x1] = False
    vals = gray[bg]
    if vals.size < 60:
        region0 = gray[y0:y1, x0:x1]
        bg_mean = float(np.percentile(region0, 30))  # robust low anchor in-box
        bg_std = float(region0.std()) * 0.6
    else:
        bg_mean, bg_std = float(np.median(vals)), float(vals.std())

    region = gray[y0:y1, x0:x1].astype(np.float32)
    thr = bg_mean + k * max(bg_std, 1.0)
    mask: Any = (region > thr).astype(np.uint8)  # cv2 morphology returns Mat | ndarray
    if mask.sum() == 0:
        thr = bg_mean + 0.8 * bg_std  # generous fallback
        mask = (region > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    full = np.zeros((hgt, wid), np.uint8)
    full[y0:y1, x0:x1] = mask
    area_frac = float(mask.sum()) / max((x1 - x0) * (y1 - y0), 1)
    return full, round(area_frac, 4)
