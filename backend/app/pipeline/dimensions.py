"""Object dimension estimation.

Across-track scale is derived from the sonar range and the slant-range
geometry; along-track scale requires ping spacing / tow speed metadata that
a single still image does not provide, so it is honestly reported as *not
estimable* rather than guessed.  Height comes from the acoustic shadow
geometry when available.
"""

from __future__ import annotations

from ..schemas import Dimensions
from . import geolocate


def estimate_dimensions(
    *,
    box: dict,
    image_height: int,
    range_m: float | None,
    altitude_m: float | None,
    height_estimate_m: float | None,
) -> Dimensions:
    if not range_m or range_m <= 0:
        return Dimensions(estimable=False, note="sonar range missing — dimensions not estimable")

    cy = box["y"] + box["h"] / 2.0
    scale = geolocate.across_track_scale(cy, image_height, float(range_m), altitude_m or 0.0)
    width_m = box["w"] * scale
    return Dimensions(
        estimable=True,
        width_m=round(width_m, 2),
        length_m=None,
        height_m=height_estimate_m,
        note="across-track scale from slant-range geometry; along-track scale not estimable from a single still",
    )
