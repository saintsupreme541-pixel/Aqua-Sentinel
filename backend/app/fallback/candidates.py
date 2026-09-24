"""Deterministic sonar-heuristic candidate generator (fallback detector).

Two complementary cues, combined:

1. **DoG blob response** — a Difference-of-Gaussians (scales ~3/11 px) fires
   on compact bright blobs (targets) and is near-zero on smooth seabed,
   ripples and speckle (which DoG suppresses at these scales).
2. **Row-CFAR confirmation** — pixels must also exceed the per-row
   median + k·MAD of their row (rows are range slices; the robust per-row
   statistics avoid the classic self-masking of windowed CFAR).

Seeds are dilated and closed into connected components, then filtered by
size, shape (no thin horizontal bands such as wall/waterline echoes) and
seed density (a real blob fills its box with seeds; a dilated thin edge
does not).  Candidate scores are deliberately conservative (0.35–0.65):
the verification stages decide the final confidence.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

MIN_AREA = 250
MAX_AREA_FRAC = 0.30
MIN_DIM = 10
MAX_ASPECT = 12.0
DOG_K = 4.5
Z_CONFIRM = 1.5
SEED_DENSITY_MIN = 0.22


def detect_heuristic(gray: np.ndarray, meta: dict[str, Any] | None = None) -> list[dict]:
    meta = meta or {}
    hgt, wid = gray.shape
    g = gray.astype(np.float32)

    # --- cue 1: DoG blob response -------------------------------------------
    g1 = cv2.GaussianBlur(g, (0, 0), 2.5)
    g2 = cv2.GaussianBlur(g, (0, 0), 10.0)
    dog = g1 - g2
    dog_mad = float(np.median(np.abs(dog))) * 1.4826 + 1e-3
    seeds = dog > DOG_K * dog_mad

    # --- cue 2: row-CFAR confirmation ---------------------------------------
    gg = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    row_med = np.median(gg, axis=1, keepdims=True)
    row_sig = np.clip(1.4826 * np.median(np.abs(gg - row_med), axis=1, keepdims=True), 2.0, None)
    zmap = (gg - row_med) / row_sig
    seeds &= zmap > Z_CONFIRM

    m: Any = seeds.astype(np.uint8)  # cv2 morphology returns Mat | ndarray; keep dtype-agnostic
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    dets: list[dict] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA or area > MAX_AREA_FRAC * hgt * wid:
            continue
        x, y, w, h = (
            int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)
        )
        if w < MIN_DIM or h < MIN_DIM:
            continue
        if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
            continue  # thin horizontal band (wall / waterline echo)

        comp = labels == i
        seed_density = float(seeds[y : y + h, x : x + w][comp[y : y + h, x : x + w]].mean())
        if seed_density < SEED_DENSITY_MIN:
            continue

        comp_mask = comp[y : y + h, x : x + w].astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1.0)
        if solidity < 0.25:
            continue

        peak = float(dog[y : y + h, x : x + w].max())
        salience = min(1.0, peak / (6.0 * dog_mad))
        sparse_frac = bright_sparse_frac(gray, {"x": float(x), "y": float(y), "w": float(w), "h": float(h)})

        # Untyped debris by default: the heuristic detector cannot reliably
        # tell a net from a can.  `bright_sparse_frac` is recorded here and
        # the net-candidate re-tag happens in the classify stage, where the
        # same metric is combined with the artificial-probability prior.
        score = min(0.68, 0.32 + 0.36 * salience)
        dets.append(
            {
                "box": {"x": float(x), "y": float(y), "w": float(w), "h": float(h)},
                "class": "debris",
                "score": round(score, 3),
                "raw": {
                    "solidity": round(solidity, 3),
                    "salience": round(salience, 3),
                    "z_peak": round(float(zmap[y : y + h, x : x + w].max()), 2),
                    "sparse_bright_frac": round(sparse_frac, 3),
                    "seed_density": round(seed_density, 3),
                },
            }
        )
    return dets


def bright_sparse_frac(gray: np.ndarray, box: dict, k: float = 3.0) -> float:
    """Fraction of the box pixels far above the box-local background.

    Filament tangles (ghost nets) are *sparse*: a modest fraction of the box
    is extremely bright, the rest is seabed.  Solid targets (pipes, cans,
    wreck plating) fill most of their box at one brightness level, so this
    fraction is ~0 there (the box median sits inside the target).
    """
    hgt, wid = gray.shape
    x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
    x1, y1 = min(wid, int(box["x"] + box["w"])), min(hgt, int(box["y"] + box["h"]))
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    if crop.size < 200:
        return 0.0
    med = float(np.median(crop))
    mad = float(np.median(np.abs(crop - med)))
    thr = med + k * max(1.4826 * mad, 4.0)
    return float((crop > thr).mean())
