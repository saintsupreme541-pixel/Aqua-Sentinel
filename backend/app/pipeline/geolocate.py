"""Sonar geometry and geolocation.

Side-scan geometry: the sonar is towed at altitude ``h``; a target at slant
range ``R_s`` lies at ground range ``R_g = sqrt(R_s^2 - h^2)`` from the track
line, abeam on either the port or starboard side.  The object bearing is
``heading +/- 90 deg`` and the location follows by haversine destination.

Uncertainty is estimated with a small Monte Carlo over the survey metadata
errors (GPS, heading, range, altitude) and summarised as a 2-sigma error
ellipse in metres (for display on the marine intelligence map).

The system NEVER fabricates coordinates: if required metadata is missing the
result is ``known=False``/``status="unknown"`` with an explanatory note.

Phase 3 provenance: every result distinguishes OBSERVED metadata (operator/
device supplied) from DERIVED quantities (bearing, ground range, position).
A derived position is always ``status="approximate"`` — never exact, never
survey-grade, never full 3D reconstruction.  The frame's own position is a
*reference*, not the object position; the offset comes from sonar geometry.

Conventions (documented in docs/geolocation.md):

- heading: degrees, 0–360, normalized (invalid/non-finite rejected)
- sonar side: ``starboard`` = heading + 90°, ``port`` = heading − 90°
- slant range: sensor→object straight line; ground range = sqrt(sl²−alt²)
  (flat-seabed).  Slant range is NEVER reported as horizontal distance.
"""

from __future__ import annotations

import math
from typing import Any, overload

import numpy as np

EARTH_RADIUS_M = 6371008.8
DEG_M = 111320.0  # metres per degree of latitude (approx)


def _finite_float(value: Any) -> float | None:
    """The value as a finite float, or None (missing/NaN/inf/non-numeric)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def normalize_heading(value: Any) -> float | None:
    """Normalize a heading to the documented 0–360° convention.

    Returns ``None`` for missing/non-finite/non-numeric values — invalid
    metadata is never silently wrapped into a plausible bearing.
    """
    v = _finite_float(value)
    return None if v is None else v % 360.0


def validate_nav(meta: dict) -> tuple[dict, list[str]]:
    """Validate observed navigation metadata (spec §15).

    Returns ``(clean, issues)``: ``clean`` holds only values that are both
    present and valid; ``issues`` names every missing/invalid field and is
    surfaced verbatim in the provenance.  Invalid input can never yield an
    apparently-valid coordinate.
    """
    issues: list[str] = []
    clean: dict[str, Any] = {}

    lat, lon = _finite_float(meta.get("lat")), _finite_float(meta.get("lon"))
    if lat is None or lon is None:
        issues.append("GPS unavailable in metadata")
    elif not (-90.0 <= lat <= 90.0):
        issues.append("latitude outside valid range [-90, 90]")
    elif not (-180.0 <= lon <= 180.0):
        issues.append("longitude outside valid range [-180, 180]")
    else:
        clean["lat"], clean["lon"] = lat, lon

    heading = normalize_heading(meta.get("heading_deg"))
    if heading is None:
        issues.append("heading missing or invalid")
    else:
        clean["heading_deg"] = heading

    alt = _finite_float(meta.get("altitude_m"))
    if alt is None:
        issues.append("altitude missing")
    elif alt <= 0:
        issues.append("altitude invalid (must be finite > 0)")
    else:
        clean["altitude_m"] = alt

    side = meta.get("side")
    if side not in ("port", "starboard"):
        issues.append("sonar side not port/starboard")
    else:
        clean["side"] = side

    rng = _finite_float(meta.get("range_m"))
    if rng is not None:
        if rng < 0:
            issues.append("sonar range invalid (must be finite >= 0)")
        else:
            clean["range_m"] = rng

    layback = _finite_float(meta.get("layback") or meta.get("layback_m"))
    if layback is not None:
        if layback < 0:
            issues.append("layback invalid (must be >= 0)")
        else:
            clean["layback_m"] = layback

    roll = _finite_float(meta.get("roll") or meta.get("roll_deg"))
    if roll is not None:
        if not (-45.0 <= roll <= 45.0):
            issues.append("roll angle outside valid range [-45, 45] deg")
        else:
            clean["roll_deg"] = roll

    pitch = _finite_float(meta.get("pitch") or meta.get("pitch_deg"))
    if pitch is not None:
        if not (-45.0 <= pitch <= 45.0):
            issues.append("pitch angle outside valid range [-45, 45] deg")
        else:
            clean["pitch_deg"] = pitch

    accuracy = _finite_float(meta.get("accuracy_m") or meta.get("accuracy") or meta.get("gps_accuracy"))
    if accuracy is not None:
        if accuracy <= 0:
            issues.append("accuracy invalid (must be > 0)")
        else:
            clean["accuracy_m"] = accuracy

    return clean, issues


def nav_inputs(meta: dict) -> dict[str, bool]:
    """Which observed navigation inputs exist (validity-checked), for provenance."""
    clean, _ = validate_nav(meta)
    return {
        "latitude": "lat" in clean,
        "longitude": "lon" in clean,
        "heading_deg": "heading_deg" in clean,
        "sonar_side": "side" in clean,
        "altitude_m": "altitude_m" in clean,
        "sonar_range_m": "range_m" in clean,
        "layback_m": "layback_m" in clean,
        "roll_deg": "roll_deg" in clean,
        "pitch_deg": "pitch_deg" in clean,
        "accuracy_m": "accuracy_m" in clean,
    }


# Heuristic floor for multi-frame location agreement (docs/geolocation.md):
# a target whose fixes spread over more than max(2×mean uncertainty, this)
# is flagged ``inconsistent``.  A documented prototype heuristic, not a
# scientifically calibrated threshold.
LOCATION_CONSISTENCY_BASE_M = 20.0


def location_consistency(positions: list[tuple[float, float]], uncertainties: list[float | None] | None = None) -> dict:
    """Do a target's derived locations agree? (spec §17/§18.)

    Distinct from *association* (§18): association asks "same object?",
    this asks "do its located observations agree spatially?".  Dispersion
    is the max pairwise haversine distance; with fewer than two located
    observations the honest answer is ``insufficient_data`` — never zero.
    """
    n = len(positions)
    if n < 2:
        return {
            "available": False,
            "status": "insufficient_data",
            "n": n,
            "dispersion_m": None,
            "score": None,
            "threshold_m": None,
        }
    disp = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            disp = max(
                disp,
                haversine_distance(positions[i][0], positions[i][1], positions[j][0], positions[j][1]),
            )
    uncs = [u for u in (uncertainties or []) if u is not None]
    mean_unc = sum(uncs) / len(uncs) if uncs else None
    threshold = (
        max(2.0 * mean_unc, LOCATION_CONSISTENCY_BASE_M) if mean_unc is not None else LOCATION_CONSISTENCY_BASE_M
    )
    return {
        "available": True,
        "status": "consistent" if disp <= threshold else "inconsistent",
        "n": n,
        "dispersion_m": round(disp, 1),
        "score": round(max(0.0, 1.0 - disp / threshold), 3),
        "threshold_m": round(threshold, 1),
    }


def haversine_destination(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    lat1, lon1 = math.radians(lat), math.radians(lon)
    brg = math.radians(bearing_deg)
    d = distance_m / EARTH_RADIUS_M
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brg))
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def geodesic_destination(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """WGS84 ellipsoidal direct geodesic calculation for high-precision offset positioning."""
    if distance_m <= 0:
        return lat, lon
    try:
        a = 6378137.0
        f = 1.0 / 298.257223563
        b = a * (1.0 - f)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        alpha1 = math.radians(bearing_deg)

        tanU1 = (1.0 - f) * math.tan(lat1)
        cosU1 = 1.0 / math.sqrt(1.0 + tanU1 * tanU1)
        sinU1 = tanU1 * cosU1

        sigma1 = math.atan2(tanU1, math.cos(alpha1))
        sinAlpha = cosU1 * math.sin(alpha1)
        cosSqAlpha = 1.0 - sinAlpha * sinAlpha

        uSq = cosSqAlpha * (a * a - b * b) / (b * b) if cosSqAlpha != 0 else 0.0
        A = 1.0 + uSq / 16384.0 * (4096.0 + uSq * (-768.0 + uSq * (320.0 - 175.0 * uSq)))
        B = uSq / 1024.0 * (256.0 + uSq * (-128.0 + uSq * (74.0 - 47.0 * uSq)))

        sigma = distance_m / (b * A)
        sigmaP = 2.0 * math.pi

        cos2SigmaM = 0.0
        sinSigma = 0.0
        cosSigma = 0.0

        for _ in range(100):
            if abs(sigma - sigmaP) <= 1e-12:
                break
            cos2SigmaM = math.cos(2.0 * sigma1 + sigma)
            sinSigma = math.sin(sigma)
            cosSigma = math.cos(sigma)
            deltaSigma = B * sinSigma * (
                cos2SigmaM
                + B / 4.0 * (
                    cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)
                    - B / 6.0 * cos2SigmaM * (-3.0 + 4.0 * sinSigma * sinSigma) * (-3.0 + 4.0 * cos2SigmaM * cos2SigmaM)
                )
            )
            sigmaP = sigma
            sigma = distance_m / (b * A) + deltaSigma

        tmp = sinU1 * sinSigma - cosU1 * cosSigma * math.cos(alpha1)
        lat2 = math.atan2(
            sinU1 * cosSigma + cosU1 * sinSigma * math.cos(alpha1),
            (1.0 - f) * math.sqrt(sinAlpha * sinAlpha + tmp * tmp),
        )
        lambda_val = math.atan2(sinSigma * math.sin(alpha1), cosU1 * cosSigma - sinU1 * sinSigma * math.cos(alpha1))
        C = f / 16.0 * cosSqAlpha * (4.0 + f * (4.0 - 3.0 * cosSqAlpha))
        L = lambda_val - (1.0 - C) * f * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM))
        )

        lon2 = lon1 + L
        res_lat = math.degrees(lat2)
        res_lon = (math.degrees(lon2) + 180.0) % 360.0 - 180.0
        if math.isfinite(res_lat) and math.isfinite(res_lon):
            return res_lat, res_lon
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return haversine_destination(lat, lon, bearing_deg, distance_m)


def fused_target_location(located_observations: list[dict]) -> dict:
    """Optimal Inverse-Variance Weighted Location Fusion across multi-pass target observations.

    Combines multiple located observations of a target into a single optimal
    fused coordinate estimate with reduced uncertainty.
    """
    if not located_observations:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "lat": None,
            "lon": None,
            "uncertainty_m": None,
            "ellipse": None,
            "n_observations": 0,
            "note": "no located observations available for target location fusion",
        }

    valid_obs = [obs for obs in located_observations if obs.get("lat") is not None and obs.get("lon") is not None]
    if not valid_obs:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "lat": None,
            "lon": None,
            "uncertainty_m": None,
            "ellipse": None,
            "n_observations": 0,
            "note": "no observation had valid coordinates",
        }

    if len(valid_obs) == 1:
        obs = valid_obs[0]
        return {
            "available": True,
            "status": "DERIVED",
            "lat": obs["lat"],
            "lon": obs["lon"],
            "uncertainty_m": obs.get("uncertainty_m"),
            "ellipse": obs.get("ellipse"),
            "n_observations": 1,
            "fusion_method": "single_observation_pass_through",
            "weights": [1.0],
            "note": "single located observation carried directly",
        }

    weights = []
    for obs in valid_obs:
        unc = obs.get("uncertainty_m")
        if unc is None or unc <= 0:
            unc = 10.0
        w = 1.0 / (max(unc, 0.5) ** 2)
        weights.append(w)

    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    fused_lat = sum(w * obs["lat"] for w, obs in zip(norm_weights, valid_obs))
    fused_lon = sum(w * obs["lon"] for w, obs in zip(norm_weights, valid_obs))
    fused_unc = round(1.0 / math.sqrt(total_w), 2)

    smax = round(max(fused_unc * 1.2, 0.5), 1)
    smin = round(max(fused_unc * 0.8, 0.3), 1)
    fused_ellipse = {"semi_major_m": smax, "semi_minor_m": smin, "rotation_deg": 0.0}

    return {
        "available": True,
        "status": "DERIVED",
        "lat": round(fused_lat, 7),
        "lon": round(fused_lon, 7),
        "uncertainty_m": fused_unc,
        "ellipse": fused_ellipse,
        "n_observations": len(valid_obs),
        "fusion_method": "inverse_variance_weighted_least_squares",
        "weights": [round(w, 4) for w in norm_weights],
        "note": f"optimal multi-pass location fusion across {len(valid_obs)} observations (uncertainty reduced to {fused_unc:.1f} m)",
    }


@overload
def slant_to_ground(slant_range_m: float, altitude_m: float) -> float: ...


@overload
def slant_to_ground(slant_range_m: np.ndarray, altitude_m: np.ndarray) -> np.ndarray: ...


@overload
def slant_to_ground(slant_range_m: np.ndarray, altitude_m: float) -> np.ndarray: ...


def slant_to_ground(slant_range_m: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """Ground range from slant range and altitude (flat-seabed assumption).

    Accepts scalars or numpy arrays (vectorised row-by-row geometry).
    """
    sl = np.asarray(slant_range_m, dtype=float)
    alt = np.asarray(altitude_m, dtype=float)
    g = np.sqrt(np.clip(sl**2 - alt**2, 0.0, None))
    if g.ndim == 0:
        return float(g)
    return g


def pixel_to_slant(row: float, image_height: int, range_m: float) -> float:
    """Map an image row to slant range (sonar at the top edge, row 0)."""
    if image_height <= 1:
        return 0.0
    return range_m * max(0.0, min(1.0, row / (image_height - 1)))


def ground_range_at(row: float, image_height: int, range_m: float, altitude_m: float) -> float:
    """Ground range (m) at an image row after slant-range correction."""
    slant = pixel_to_slant(row, image_height, range_m)
    if altitude_m is None or altitude_m <= 0:
        return slant
    return slant_to_ground(slant, altitude_m)


def across_track_scale(row: float, image_height: int, range_m: float, altitude_m: float) -> float:
    """Metres per pixel in the across-track direction at a given row."""
    r0 = ground_range_at(row, image_height, range_m, altitude_m)
    r1 = ground_range_at(row + 1.0, image_height, range_m, altitude_m)
    return max(r1 - r0, 1e-6)


def unknown(note: str, provenance: dict | None = None) -> dict:
    """Honest "location unavailable" result: unknown ≠ zero, ≠ low confidence."""
    return {
        "known": False,
        "status": "unknown",
        "lat": None,
        "lon": None,
        "uncertainty_m": None,
        "ellipse": None,
        "note": note,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Geolocation v2 — provenance state model (VERIFIED/DERIVED/MANUAL/
# VESSEL_ONLY/UNAVAILABLE/INVALID).  The legacy engine above is unchanged;
# these helpers wrap its results and normalize statuses across the API.
# ---------------------------------------------------------------------------

GEO_STATUSES = ("VERIFIED", "DERIVED", "MANUAL", "VESSEL_ONLY", "UNAVAILABLE", "INVALID")

#: legacy → v2 status normalization (old persisted rows keep working)
STATUS_MAP = {
    "approximate": "DERIVED",
    "known": "VERIFIED",
    "unknown": "UNAVAILABLE",
}


def normalize_status(status: str | None) -> str:
    """Any persisted status → the v2 state model (never silently upgrades:
    legacy ``approximate`` becomes DERIVED, ``unknown`` becomes UNAVAILABLE)."""
    if not status:
        return "UNAVAILABLE"
    s = str(status).upper()
    if s in GEO_STATUSES:
        return s
    return STATUS_MAP.get(str(status).lower(), "UNAVAILABLE")


def manual_geolocation(
    lat: float,
    lon: float,
    *,
    note: str = "",
    reference_frame_id: str | None = None,
) -> dict:
    """An explicit user-created georeference (§16): status MANUAL, source
    'user_supplied'.  Manual coordinates are never presented as GNSS."""
    return {
        "known": True,
        "status": "MANUAL",
        "lat": lat,
        "lon": lon,
        "uncertainty_m": None,
        "ellipse": None,
        "note": note or "manually supplied by the operator (not sensor-derived)",
        "provenance": {
            "kind": "manual",
            "status": "MANUAL",
            "source": "user_supplied",
            "reference_frame_id": reference_frame_id,
            "inputs": {"latitude": True, "longitude": True},
            "derived": {},
            "assumptions": [],
            "issues": [],
            "note": "operator-created georeference — never GNSS-derived",
        },
    }


def vessel_only(note: str, frame_position: dict | None, provenance: dict | None = None) -> dict:
    """Vessel position exists but no target-level fix can be justified (§1).

    The frame/vessel position is carried separately as ``frame_position``
    (clearly labelled vessel, never merged into the target coordinates).
    """
    return {
        "known": False,
        "status": "VESSEL_ONLY",
        "lat": None,
        "lon": None,
        "frame_position": frame_position,
        "uncertainty_m": None,
        "ellipse": None,
        "note": note,
        "provenance": provenance,
    }


def invalid(note: str, issues: list[str], provenance: dict | None = None) -> dict:
    """Navigation exists but failed validation (§1) — rejected, not repaired."""
    prov = {
        **(provenance or {"kind": "invalid", "inputs": {}, "derived": {}, "assumptions": []}),
        "kind": "invalid",
        "status": "INVALID",
        "issues": issues,
    }
    return {
        "known": False,
        "status": "INVALID",
        "lat": None,
        "lon": None,
        "uncertainty_m": None,
        "ellipse": None,
        "note": note,
        "provenance": prov,
    }


def locate_detection(
    *,
    meta: dict,
    slant_range_m: float | None,
    image_height: int = 0,
    row: float | None = None,
    reference_frame_id: str | None = None,
) -> dict:
    """Geolocate a detection from survey metadata and sonar geometry.

    ``slant_range_m`` may be given directly, or computed from ``row`` and
    ``meta["range_m"]``.  The result carries an explicit ``status``
    (``approximate``/``unknown``) and a structured ``provenance`` block
    separating observed inputs from derived quantities (Phase 3).
    """
    clean, issues = validate_nav(meta)
    inputs = nav_inputs(meta)
    unavailable_prov: dict[str, Any] = {
        "kind": "unavailable",
        "status": "unknown",
        "source": None,
        "reference_frame_id": reference_frame_id,
        "inputs": inputs,
        "derived": {},
        "assumptions": [],
        "issues": issues,
        "note": "no coordinate produced — missing or invalid navigation metadata",
    }

    required = ("lat", "lon", "heading_deg", "side", "altitude_m")
    if any(k not in clean for k in required):
        return unknown("; ".join(issues) or "navigation metadata unavailable", unavailable_prov)

    altitude = float(clean["altitude_m"])
    heading = float(clean["heading_deg"])
    side = str(clean["side"])

    if slant_range_m is not None:
        slant = float(slant_range_m)
        if not math.isfinite(slant) or slant <= 0:
            unavailable_prov["issues"] = [*issues, "slant range invalid (must be finite > 0)"]
            return unknown("slant range invalid", unavailable_prov)
    else:
        if row is None or image_height <= 0 or "range_m" not in clean:
            unavailable_prov["issues"] = [
                *issues,
                "slant range unavailable (no direct value; row→slant conversion needs image height + sonar range)",
            ]
            return unknown("slant range unavailable", unavailable_prov)
        slant = pixel_to_slant(float(row), image_height, float(clean["range_m"]))
        if not math.isfinite(slant) or slant <= 0:
            unavailable_prov["issues"] = [*issues, "derived slant range is zero/invalid at this row"]
            return unknown("slant range unavailable", unavailable_prov)

    # 1. Towfish Layback Correction
    layback = clean.get("layback_m")
    derived_info: dict[str, Any] = {}
    assumptions: list[str] = [
        "flat-seabed sonar geometry",
        "object abeam of the track: starboard = heading+90°, port = heading−90°",
    ]

    if layback is not None and layback > 0:
        sensor_lat, sensor_lon = geodesic_destination(
            float(clean["lat"]), float(clean["lon"]), (heading + 180.0) % 360.0, layback
        )
        assumptions.append(f"towfish layback correction applied ({layback:.1f} m behind vessel)")
        derived_info["layback_applied_m"] = round(layback, 2)
    else:
        sensor_lat, sensor_lon = float(clean["lat"]), float(clean["lon"])

    # 2. Transducer Roll & Pitch Attitude Compensation
    roll_deg = clean.get("roll_deg")
    pitch_deg = clean.get("pitch_deg")
    effective_altitude = altitude
    along_track_offset_m = 0.0

    if roll_deg is not None and abs(roll_deg) > 0.01:
        roll_rad = math.radians(roll_deg)
        effective_altitude = max(0.1, altitude * math.cos(roll_rad))
        derived_info["roll_deg"] = round(roll_deg, 2)
        derived_info["effective_altitude_m"] = round(effective_altitude, 2)
        assumptions.append(f"transducer roll compensation applied ({roll_deg:.1f}° roll)")

    if pitch_deg is not None and abs(pitch_deg) > 0.01:
        pitch_rad = math.radians(pitch_deg)
        along_track_offset_m = altitude * math.tan(pitch_rad)
        derived_info["pitch_deg"] = round(pitch_deg, 2)
        derived_info["along_track_offset_m"] = round(along_track_offset_m, 2)
        assumptions.append(f"transducer pitch compensation applied ({pitch_deg:.1f}° pitch)")

    if slant <= effective_altitude:
        # Physically impossible for an abeam seabed object: the slant range
        # never leaves the sonar plane, so there is NO valid horizontal
        # offset.  Refuse rather than collapse the fix onto the frame
        # position (which would fabricate a location).
        unavailable_prov["issues"] = [
            *issues,
            f"slant range ({slant:.2f} m) <= effective altitude ({effective_altitude:.2f} m) — no physically valid ground offset",
        ]
        return unknown("geometry invalid: slant range does not exceed altitude", unavailable_prov)

    ground = slant_to_ground(slant, effective_altitude)
    bearing = (heading + (90.0 if side == "starboard" else -90.0)) % 360.0

    # Primary offset from transducer
    lat, lon = geodesic_destination(sensor_lat, sensor_lon, bearing, ground)

    # Secondary offset from pitch along-track displacement
    if abs(along_track_offset_m) > 0.01:
        pitch_bearing = heading if along_track_offset_m > 0 else (heading + 180.0) % 360.0
        lat, lon = geodesic_destination(lat, lon, pitch_bearing, abs(along_track_offset_m))

    gps_err = max(0.5, float(clean.get("accuracy_m", 5.0)))
    unc = _monte_carlo_uncertainty(
        vessel_lat=sensor_lat,
        vessel_lon=sensor_lon,
        heading=heading,
        side=side,
        slant_range_m=slant,
        altitude_m=effective_altitude,
        gps_err_m=gps_err,
    )
    if "accuracy_m" in clean:
        derived_info["gps_accuracy_used_m"] = round(gps_err, 1)

    provenance: dict[str, Any] = {
        "kind": "derived",
        "status": "approximate",
        "source": "frame_navigation_plus_sonar_geometry",
        "reference_frame_id": reference_frame_id,
        "inputs": inputs,
        "derived": {
            "bearing_deg": round(bearing, 1),
            "slant_range_m": round(slant, 2),
            "ground_range_m": round(ground, 2),
            **derived_info,
        },
        "assumptions": assumptions,
        "issues": [],
        "note": (
            "frame position ≠ object position — the offset is derived from sonar "
            "geometry; result is approximate, never survey-grade, never 3D reconstruction"
        ),
    }
    return {
        "known": True,
        "status": "approximate",
        "lat": lat,
        "lon": lon,
        "uncertainty_m": unc["radius_m"],
        "ellipse": unc["ellipse"],
        "note": "estimated from survey metadata and sonar geometry (slant-range corrected)",
        "bearing_deg": bearing,
        "ground_range_m": ground,
        "provenance": provenance,
    }


def _monte_carlo_uncertainty(
    *,
    vessel_lat: float,
    vessel_lon: float,
    heading: float,
    side: str,
    slant_range_m: float,
    altitude_m: float,
    heading_err_deg: float = 2.0,
    gps_err_m: float = 5.0,
    range_err_frac: float = 0.05,
    altitude_err_frac: float = 0.05,
    n_samples: int = 300,
) -> dict:
    rng = np.random.default_rng(0)
    hs = heading + rng.normal(0.0, heading_err_deg, n_samples)
    gps = rng.normal(0.0, gps_err_m, (n_samples, 2))
    sl = slant_range_m * (1 + rng.normal(0.0, range_err_frac, n_samples))
    alt = altitude_m * (1 + rng.normal(0.0, altitude_err_frac, n_samples))
    ground = np.sqrt(np.clip(sl**2 - alt**2, 0, None))
    brg = np.radians((hs + (90.0 if side == "starboard" else -90.0)) % 360.0)
    # local tangent-plane coordinates (east, north) in metres
    e = ground * np.sin(brg) + gps[:, 0]
    n = ground * np.cos(brg) + gps[:, 1]
    cov = np.cov(np.stack([e, n]))
    vals, vecs = np.linalg.eigh(cov)
    smax, smin = 2.0 * np.sqrt(np.clip(vals[1], 0, None)), 2.0 * np.sqrt(np.clip(vals[0], 0, None))
    rot = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    radius = math.sqrt(smax * smin)
    return {
        "radius_m": round(radius, 1),
        "ellipse": {"semi_major_m": round(smax, 1), "semi_minor_m": round(smin, 1), "rotation_deg": round(rot, 1)},
    }


def ellipse_polygon(
    lat: float, lon: float, semi_major_m: float, semi_minor_m: float, rotation_deg: float, n: int = 36
) -> list[list[float]]:
    """Approximate the uncertainty ellipse as a GeoJSON polygon (lon/lat pairs)."""
    lon_scale = DEG_M * math.cos(math.radians(lat)) or DEG_M
    pts = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        ex, ey = semi_major_m * math.cos(ang), semi_minor_m * math.sin(ang)
        rot = math.radians(rotation_deg)
        x = ex * math.cos(rot) - ey * math.sin(rot)
        y = ex * math.sin(rot) + ey * math.cos(rot)
        pts.append([round(lon + x / lon_scale, 7), round(lat + y / DEG_M, 7)])
    pts.append(pts[0])
    return pts
