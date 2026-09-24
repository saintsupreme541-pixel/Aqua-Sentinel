"""Physics-informed acoustic-shadow analysis (sonar geometry).

This module converts *image-space* shadow evidence into *physical*
quantities using the sonar geometry available in the survey metadata.
It is deliberately conservative:

- It does **not** reconstruct 3D geometry.  On a flat-seabed assumption
  the shadow/height relation is a similar-triangles estimate::

      grazing θ = asin(altitude / slant_range)
      height h ≈ H · L_shadow_ground / R_far        (from shadow.py)

- When the metadata required by the geometry is missing the module
  reports ``metadata_sufficient = False`` and returns ``None`` for the
  physical quantities — image-based shadow evidence remains valid, and
  the fusion layer renormalizes around the missing physics signal.

- The *geometry score* is a plausibility check, not an accuracy claim:
  acoustic shadows are only well-formed within a grazing-angle window.
  Outside that window a "no shadow" result is expected physics, not a
  suspicious detection.
"""

from __future__ import annotations

import math

from ..schemas import PhysicsInfo
from . import geolocate

# Grazing-angle window in which acoustic shadows are physically well formed.
# Below GRAZING_MIN the beam skims a nearly-flat seabed (shadows stretch and
# smear); above GRAZING_MAX the beam is too vertical for a shadow to form.
GRAZING_MIN_DEG = 8.0
GRAZING_MAX_DEG = 65.0


def grazing_angle_deg(altitude_m: float, slant_range_m: float) -> float | None:
    """Grazing angle from the sonar-to-seabed geometry (flat-seabed far field)."""
    if altitude_m is None or slant_range_m is None:
        return None
    if altitude_m <= 0 or slant_range_m <= 0:
        return None
    if slant_range_m < altitude_m:  # geometrically impossible for a seabed return
        return None
    s = min(1.0, altitude_m / slant_range_m)
    return math.degrees(math.asin(s))


def _geometry_score(angle_deg: float | None) -> float | None:
    """Plausibility that the sonar geometry can produce/interpret shadows.

    1.0 inside the well-formed window, decaying linearly to 0 at 0.5×/1.5×
    the window edges.  ``None`` when the angle itself is unknowable.
    """
    if angle_deg is None:
        return None
    lo, hi = GRAZING_MIN_DEG, GRAZING_MAX_DEG
    if angle_deg < lo:
        return max(0.0, 1.0 - (lo - angle_deg) / (0.5 * lo))
    if angle_deg > hi:
        return max(0.0, 1.0 - (angle_deg - hi) / (0.5 * hi))
    return 1.0


def analyze_physics(
    *,
    box: dict,
    shadow: dict,
    meta: dict,
    image_height: int,
) -> PhysicsInfo:
    """Derive approximate physical quantities for one detection.

    ``shadow`` is the ShadowInfo dict produced by ``shadow.analyze_shadow``
    (already contains the similar-triangles height estimate when the shadow
    was valid and altitude/range metadata existed).
    """
    range_m = meta.get("range_m")
    altitude_m = meta.get("altitude_m")
    row = float(box["y"]) + float(box["h"]) / 2.0

    # slant range at the object row (SSS waterfall: row ↔ slant range)
    slant_m: float | None = None
    if range_m and range_m > 0 and image_height:
        try:
            slant_m = geolocate.pixel_to_slant(row, image_height, float(range_m))
        except Exception:  # noqa: BLE001 — geometry is best-effort, never fatal
            slant_m = None

    angle = grazing_angle_deg(altitude_m, slant_m) if slant_m is not None else None
    angle = round(angle, 1) if angle is not None else None
    score = _geometry_score(angle)
    score = round(score, 3) if score is not None else None

    shadow_valid = bool(shadow.get("valid"))
    shadow_height = shadow.get("height_estimate_m")

    sufficient = (
        range_m is not None
        and range_m > 0
        and altitude_m is not None
        and altitude_m > 0
        and slant_m is not None
        and shadow_valid
        and shadow_height is not None
    )

    notes: list[str] = []
    if range_m is None or not range_m:
        notes.append("sonar range missing")
    if altitude_m is None or altitude_m <= 0:
        notes.append("sonar altitude missing")
    if slant_m is None and (range_m and range_m > 0):
        notes.append("row→slant-range conversion unavailable")
    if not shadow_valid:
        notes.append("no valid acoustic shadow — height not estimable")
    if sufficient and angle is not None:
        notes.append(
            f"grazing {angle:.0f}° within shadow window "
            f"[{GRAZING_MIN_DEG:.0f}°, {GRAZING_MAX_DEG:.0f}°] — flat-seabed similar-triangles estimate"
        )
    elif angle is not None:
        notes.append(f"grazing {angle:.0f}° outside/edge of shadow window — treat height with caution")

    return PhysicsInfo(
        metadata_sufficient=sufficient,
        grazing_angle_deg=angle,
        estimated_height_m=round(float(shadow_height), 2) if sufficient and shadow_height is not None else None,
        geometry_score=score if sufficient or score is not None else None,
        note="; ".join(notes) if notes else "sonar geometry not assessable",
    )
