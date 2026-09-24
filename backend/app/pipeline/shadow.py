"""Acoustic shadow analysis.

A raised object blocks the sonar beam, leaving a dark (no-echo) region on
the far side — an *acoustic shadow*.  We treat shadows as *evidence, when
available*, not as a requirement: the geometry must be assessable (we know
which direction the sonar illuminates from) and the shadow region must be
physically plausible (contiguous, adjacent to the highlight, width-matched).

When valid, shadow length gives an object-height estimate via similar
triangles on a flat seabed::

    h / H = L_shadow / R_far      →   h = H * L_shadow / R_far

where H is the sonar altitude, L_shadow the ground shadow length and R_far
the ground range to the far end of the shadow.
"""

from __future__ import annotations

import numpy as np

from ..schemas import ShadowInfo
from . import geolocate

# intensity below background (in std units) required to count as shadow
SHADOW_STD_FACTOR = 1.5
# how far (px) below the object the shadow is allowed to start
MAX_START_GAP_PX = 6
# minimum width of the shadow run relative to the object width
MIN_WIDTH_FRAC = 0.5


def _row_profile_region(gray: np.ndarray, box: dict, direction: str, max_len: int) -> np.ndarray | None:
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    hgt, wid = gray.shape
    x0, x1 = max(0, x), min(wid, x + w)
    if direction == "down":
        y0, y1 = min(hgt, y + h), min(hgt, y + h + max_len)
    elif direction == "up":
        y0, y1 = max(0, y - max_len), max(0, y)
    elif direction in ("right",):
        raise NotImplementedError  # rotate caller's responsibility; we support vertical
    else:
        return None
    if y1 <= y0 or x1 <= x0:
        return None
    return gray[y0:y1, x0:x1]


def analyze_shadow(
    gray: np.ndarray,
    box: dict,
    *,
    direction: str | None = "down",
    range_m: float | None = None,
    altitude_m: float | None = None,
    image_height: int | None = None,
) -> tuple[ShadowInfo, float | None]:
    """Analyse the acoustic shadow for one detection.

    Returns ``(ShadowInfo, evidence_value)`` where the evidence value is
    None when shadow evidence is *not assessable* (distinct from "no
    shadow", which scores 0.0).
    """
    hgt, wid = gray.shape
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    x0, x1 = max(0, x), min(wid, x + w)
    y0, y1 = max(0, y), min(hgt, y + h)

    if x1 <= x0 or y1 <= y0:
        return ShadowInfo(note="invalid box"), None

    # background statistics from a ring around the box (avoid contaminating with the object)
    bg_mask = np.ones_like(gray, dtype=bool)
    bg_mask[y0:y1, x0:x1] = False
    pad = max(4, int(0.5 * max(w, h)))
    yb0, yb1 = max(0, y0 - pad), min(hgt, y1 + pad)
    xb0, xb1 = max(0, x0 - pad), min(wid, x1 + pad)
    bg_mask[yb0:yb1, xb0:xb1] = False
    bg = gray[~bg_mask] if bg_mask.sum() < 200 else gray[bg_mask]
    bg_mean = float(np.mean(bg))
    bg_std = float(np.std(bg)) + 1e-6

    if direction is None or direction in ("left", "right"):
        return ShadowInfo(note="shadow direction unknown — evidence not assessable"), None

    max_len = max(8, int(4 * h))
    region = _row_profile_region(gray, box, direction, max_len)
    if region is None or region.size == 0:
        return ShadowInfo(note="no shadow region available"), None

    profile = region.mean(axis=1) if direction in ("down", "up") else region.mean(axis=0)
    n_rows = len(profile)
    thr = bg_mean - SHADOW_STD_FACTOR * bg_std
    dark = profile < thr

    # find first contiguous run, allowing a small gap after the object edge
    start_gap = 0
    while start_gap < n_rows and profile[start_gap] >= bg_mean - 0.75 * bg_std:
        start_gap += 1
    run_start = start_gap
    run_end = run_start
    while run_end < n_rows and dark[run_end]:
        run_end += 1
    length_px = float(run_end - run_start)
    valid = False
    if start_gap <= MAX_START_GAP_PX and length_px >= 2 and _width_match(region, run_start, run_end, w, direction):
        valid = True

    info = ShadowInfo(available=True, valid=valid, length_px=length_px if valid else None)
    evidence: float | None = 0.0
    if not valid:
        info.note = "no valid acoustic shadow detected (flat or buried object, or geometry mismatch)"
    else:
        # strength: long shadows relative to object height are stronger evidence
        strength = min(1.0, length_px / max(0.5 * h, 1.0))
        evidence = 0.5 + 0.4 * strength
        info.note = "valid acoustic shadow detected"

    # height estimate via similar triangles
    if valid and range_m and altitude_m and image_height and altitude_m > 0:
        slant_far = geolocate.pixel_to_slant(float(y1 + run_end), image_height, float(range_m))
        slant_near = geolocate.pixel_to_slant(float(y1 + run_start), image_height, float(range_m))
        r_far = geolocate.slant_to_ground(slant_far, altitude_m)
        r_near = geolocate.slant_to_ground(slant_near, altitude_m)
        l_ground = max(r_far - r_near, 1e-6)
        if r_far > 0:
            height = altitude_m * l_ground / r_far
            info.length_m = round(l_ground, 2)
            info.height_estimate_m = round(height, 2)
            info.note += f"; estimated object height {height:.2f} m"
    return info, evidence


def _width_match(region: np.ndarray, run_start: int, run_end: int, obj_w: int, direction: str) -> bool:
    """Shadow run should span a similar across-track width as the object."""
    if run_end <= run_start:
        return False
    sub = region[run_start:run_end]
    dark_frac = (sub < sub.mean()).mean(axis=0 if direction in ("down", "up") else 1)
    width_span = int(np.sum(dark_frac > 0.5))
    return width_span >= MIN_WIDTH_FRAC * obj_w


def default_direction(meta: dict) -> str | None:
    """Resolve the illumination direction from metadata when possible."""
    sonar_type = meta.get("sonar_type", "unknown")
    if sonar_type == "sss":
        return "down"  # sonar at top (nadir), range increasing downward
    if sonar_type == "fls":
        return "down"  # forward-looking: far range at top or bottom depending on mount
    return meta.get("shadow_direction", "down")
