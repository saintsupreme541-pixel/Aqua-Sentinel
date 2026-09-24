"""Navigation ingestion, validation and timestamp synchronization.

GEOLOCATION v2 — provenance-first architecture.  This module turns uploaded
navigation files (CSV / GeoCSV / GeoJSON / GPX) into a validated survey
track and synchronizes sonar frames to it by timestamp.

Core rules (spec §18 — data integrity):

- A coordinate enters the system ONLY from a genuine uploaded record or a
  user's explicit manual georeference.  Nothing here invents positions.
- Invalid records are REJECTED with explicit diagnostics; the system never
  silently repairs suspicious navigation (§3).
- Timestamp synchronization uses the documented methods ``exact_timestamp``,
  ``linear_interpolation`` or ``nearest`` (only when interpolation is
  impossible) and records which one produced the frame position (§4).
- CRS handling: input declared as UTM/projected is transformed to EPSG:4326
  with pyproj; WGS84 is never assumed when a source CRS is given (§9).

Supported input columns (case-insensitive; common aliases accepted):

    timestamp / time / utc / iso_time   → ts
    lat / latitude / lat_deg            → latitude
    lon / lon / lng / longitude         → longitude
    heading / hdg / course / cog        → heading_deg
    altitude / alt / depth / sonar_depth→ altitude_m
    accuracy / hdop*5 (recorded as-is)  → accuracy_m
    roll / pitch / heave, towfish_x/y/z, layback, source

GeoJSON: FeatureCollection/Geometry with Point features; properties carry
the optional fields.  GPX: ``<trkpt>/<rtept>/<wpt>`` with ``<time>`` and
``<speed>`` children parsed when present; ``ele`` is stored as altitude.
"""

from __future__ import annotations

import csv
import io
import json
import math
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Any

from pyproj import CRS, Transformer

# ---------------------------------------------------------------------------
# Public constants / contracts
# ---------------------------------------------------------------------------

SYNC_EXACT = "exact_timestamp"
SYNC_LINEAR = "linear_interpolation"
SYNC_NEAREST = "nearest"

#: Mean Earth radius (m) — matches geolocate.EARTH_RADIUS_M.
EARTH_RADIUS_M = 6371008.8

#: realistic lower bound for a surface/towed vessel over ground (m/s).  A
#: jump implying more than this flags BOTH records as suspect (§3: expose,
#: never repair).
MAX_SPEED_MS = 15.0  # ≈ 29 kn — far above any survey vessel
#: two fixes closer than this in time are considered duplicate samples (§3)
DUPLICATE_TS_EPSILON = timedelta(milliseconds=500)
#: navigation gap longer than this around a frame is reported in diagnostics
GAP_THRESHOLD = timedelta(seconds=60)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ts": ("timestamp", "time", "utc", "iso_time", "datetime", "gps_time"),
    "latitude": ("lat", "latitude", "lat_deg", "y"),
    "longitude": ("lon", "lng", "long", "longitude", "lon_deg", "x"),
    "heading_deg": ("heading", "hdg", "course", "cog", "heading_deg", "true_heading"),
    "altitude_m": ("altitude", "alt", "altitude_m", "depth", "sonar_depth", "fish_depth"),
    "accuracy_m": ("accuracy", "accuracy_m", "gps_accuracy", "hdop"),
    "source": ("source", "gps_source", "receiver"),
    "roll": ("roll", "roll_deg"),
    "pitch": ("pitch", "pitch_deg"),
    "heave": ("heave", "heave_m"),
    "towfish_x": ("towfish_x", "fish_x", "offset_x"),
    "towfish_y": ("towfish_y", "fish_y", "offset_y"),
    "towfish_z": ("towfish_z", "fish_z", "offset_z"),
    "layback": ("layback", "layback_m"),
}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Finite float or None — strings, NaN and infinities all rejected."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _parse_ts(value: Any) -> datetime | None:
    """ISO-8601 (with or without Z / offset) or unix seconds → aware UTC dt."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if not math.isfinite(v) or v < 0:
            return None
        return datetime.fromtimestamp(v, tz=UTC)
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # plain date → midnight UTC
        try:
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)  # naive timestamps are treated as UTC
    return dt.astimezone(UTC)


def _map_columns(fieldnames: Any) -> dict[str, str]:
    """Map file columns onto canonical names (case/alias tolerant, §6)."""
    mapping: dict[str, str] = {}
    lowered = {name.strip().lower(): name for name in fieldnames}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[canonical] = lowered[alias]
                break
    return mapping


# ---------------------------------------------------------------------------
# Record validation (§3)
# ---------------------------------------------------------------------------


def _validate_record(raw: dict[str, Any], *, index: int, declared_crs: str | None) -> tuple[dict[str, Any], list[str]]:
    """Validate one navigation sample.  Returns (clean, issues).

    ``clean`` is only produced when the record is fully usable; any fatal
    issue yields ``None`` and the record is rejected with reasons.
    """
    issues: list[str] = []

    lat = _num(raw.get("latitude"))
    lon = _num(raw.get("longitude"))
    if lat is None or lon is None:
        return {}, ["missing latitude/longitude"]

    # §9: transform FIRST when a source CRS is declared — projected input is
    # in metres and would wrongly fail the WGS84 range check.  WGS84 is
    # never assumed when a source CRS is explicitly provided.
    crs_note: str | None = None
    if declared_crs and not _is_wgs84(declared_crs):
        try:
            lat, lon = _to_wgs84(lat, lon, declared_crs)
            crs_note = f"transformed from {declared_crs} to EPSG:4326"
        except Exception as e:  # noqa: BLE001 — pyproj error text is diagnostic
            return {}, [f"CRS transform failed: {e}"]

    if not (-90.0 <= lat <= 90.0):
        return {}, [f"latitude out of range [-90, 90]: {lat}"]
    if not (-180.0 <= lon <= 180.0):
        return {}, [f"longitude out of range [-180, 180]: {lon}"]

    ts = _parse_ts(raw.get("ts"))
    if ts is None:
        issues.append("missing or unparseable timestamp")

    heading = _num(raw.get("heading_deg"))
    if heading is not None and (heading < 0 or heading >= 360):
        issues.append(f"heading outside [0, 360): {heading}")

    altitude = _num(raw.get("altitude_m"))
    if altitude is not None and altitude < 0:
        issues.append(f"negative altitude/depth: {altitude}")

    accuracy = _num(raw.get("accuracy_m"))
    if accuracy is not None and accuracy < 0:
        issues.append(f"negative accuracy: {accuracy}")

    clean: dict[str, Any] = {
        "ts": ts.isoformat().replace("+00:00", "Z") if ts else None,
        "latitude": round(lat, 7),
        "longitude": round(lon, 7),
        "heading_deg": heading,
        "altitude_m": altitude,
        "accuracy_m": accuracy,
        "source": raw.get("source") or "uploaded_navigation",
        "roll": _num(raw.get("roll")),
        "pitch": _num(raw.get("pitch")),
        "heave": _num(raw.get("heave")),
        "towfish_x": _num(raw.get("towfish_x")),
        "towfish_y": _num(raw.get("towfish_y")),
        "towfish_z": _num(raw.get("towfish_z")),
        "layback": _num(raw.get("layback")),
        "valid": True,
        "issues": [crs_note] if crs_note else [],
        "_index": index,
    }
    return clean, issues


def _is_wgs84(crs_string: str) -> bool:
    s = crs_string.strip().upper()
    return s in ("EPSG:4326", "WGS84", "CRS84", "OGC:CRS84", "4326")


def _to_wgs84(lat: float, lon: float, crs_string: str = "EPSG:4326") -> tuple[float, float]:
    """Transform a point to EPSG:4326.  UTM zone shorthand (e.g. 'UTM44N')
    and any proj/epsg string pyproj understands are supported (§9)."""
    s = crs_string.strip().upper().replace(" ", "")
    if s.startswith("UTM"):
        zone = int("".join(ch for ch in s if ch.isdigit())[:2])
        north = not s.endswith("S")
        crs_string = f"+proj=utm +zone={zone} +{'north' if north else 'south'} +datum=WGS84"
        transformer = Transformer.from_crs(crs_string, CRS.from_epsg(4326), always_xy=True)
        x = lon  # UTM shorthand carries easting in the lon slot
        y = lat
        lon_out, lat_out = transformer.transform(x, y)
        return float(lat_out), float(lon_out)
    transformer = Transformer.from_crs(CRS.from_user_input(crs_string), CRS.from_epsg(4326), always_xy=True)
    lon_out, lat_out = transformer.transform(lon, lat)
    return float(lat_out), float(lon_out)


# ---------------------------------------------------------------------------
# Track-level validation (§3: ordering, duplicates, jumps, gaps)
# ---------------------------------------------------------------------------


def _attach_track_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag duplicates, impossible jumps and gaps; drop jump-suspect fixes.

    A record implicated in an impossible-speed jump is marked invalid with
    the reason attached — the honest alternative to silently "fixing" the
    position.  Diagnostics always list every finding.
    """
    out = [dict(r) for r in records]
    dated = [r for r in out if r.get("_ts") is not None]
    dated.sort(key=lambda r: r["_ts"])

    for i in range(1, len(dated)):
        prev, cur = dated[i - 1], dated[i]
        dt = (cur["_ts"] - prev["_ts"]).total_seconds()
        if dt <= 0 and (cur["_ts"] - prev["_ts"]) < DUPLICATE_TS_EPSILON and _index_key(prev) == _index_key(cur):
            # duplicate timestamp pair (adjacent identical times)
            cur["issues"] = [*cur.get("issues", []), "duplicate timestamp with previous record"]
            cur["valid"] = False
        if dt > 0:
            dist = _haversine(prev["latitude"], prev["longitude"], cur["latitude"], cur["longitude"])
            speed = dist / dt
            if speed > MAX_SPEED_MS and dist > 50.0:
                # both endpoints are suspect — neither can be trusted alone
                if prev["valid"] and not any("impossible jump" in x for x in prev.get("issues", [])):
                    prev["issues"] = [
                        *prev.get("issues", []),
                        f"impossible jump: {speed:.1f} m/s to next fix ({dist:.0f} m in {dt:.0f} s)",
                    ]
                    prev["valid"] = False
                if cur["valid"] and not any("impossible jump" in x for x in cur.get("issues", [])):
                    cur["issues"] = [
                        *cur.get("issues", []),
                        f"impossible jump: {speed:.1f} m/s from previous fix ({dist:.0f} m in {dt:.0f} s)",
                    ]
                    cur["valid"] = False
    return out


def _index_key(r: dict[str, Any]) -> tuple[float, float]:
    return (r.get("latitude") or 0.0, r.get("longitude") or 0.0)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (m) — same formula as geolocate.haversine_distance;
    kept local so this module needs no pipeline import."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# File parsers → raw record dicts
# ---------------------------------------------------------------------------


def parse_csv(content: bytes) -> tuple[list[dict[str, Any]], str | None]:
    """CSV / GeoCSV (with optional WKT or lat/lon columns).  Returns
    (records, declared_crs)."""
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return [], None
    mapping = _map_columns(reader.fieldnames)
    if "latitude" not in mapping or "longitude" not in mapping:
        return [], None
    declared_crs = None
    # GeoCSV convention: a '.crs' column like "EPSG:32644" or WKT CRS row
    for name in reader.fieldnames:
        if name.strip().lower().endswith(".crs") or name.strip().lower() == "crs":
            declared_crs = (reader.fieldnames and None) or name  # column present; values read below
            break
    records: list[dict[str, Any]] = []
    for row in reader:
        crs_val = None
        if declared_crs:
            crs_val = (row.get(declared_crs) or "").strip() or None
        rec: dict[str, Any] = {"source": row.get(mapping.get("source", ""), "") or None}
        for canonical, file_col in mapping.items():
            if canonical == "source":
                continue
            rec[canonical] = row.get(file_col)
        rec["_crs"] = crs_val
        records.append(rec)
    # row-level CRS wins over a single file-level declaration
    file_crs = records[0].get("_crs") if records else None
    return records, file_crs


def parse_geojson(content: bytes) -> tuple[list[dict[str, Any]], str | None]:
    """GeoJSON FeatureCollection / Geometry — Point features only."""
    try:
        doc = json.loads(content.decode("utf-8-sig", errors="replace"))
    except ValueError:
        return [], None
    features: list[dict[str, Any]] = []
    if doc.get("type") == "FeatureCollection":
        features = doc.get("features", [])
    elif doc.get("type") == "Feature":
        features = [doc]
    elif doc.get("type") == "Point":
        features = [{"geometry": doc, "properties": {}}]
    else:
        return [], None
    crs_name = None
    crs = doc.get("crs") or {}
    if isinstance(crs, dict):
        props = crs.get("properties") or {}
        crs_name = props.get("name")
    records: list[dict[str, Any]] = []
    for f in features:
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = f.get("properties") or {}
        rec = {
            "ts": props.get("timestamp") or props.get("time") or props.get("t"),
            "latitude": coords[1],
            "longitude": coords[0],
            "heading_deg": props.get("heading") or props.get("heading_deg"),
            "altitude_m": props.get("altitude") or props.get("depth") or props.get("altitude_m"),
            "accuracy_m": props.get("accuracy") or props.get("accuracy_m"),
            "source": props.get("source") or "geojson",
            "roll": props.get("roll"),
            "pitch": props.get("pitch"),
            "heave": props.get("heave"),
            "towfish_x": props.get("towfish_x"),
            "towfish_y": props.get("towfish_y"),
            "towfish_z": props.get("towfish_z"),
            "layback": props.get("layback"),
        }
        records.append(rec)
    return records, crs_name


def parse_gpx(content: bytes) -> tuple[list[dict[str, Any]], str | None]:
    """GPX 1.0/1.1 — trkpt/rtept/wpt with optional <time>/<ele> children."""
    try:
        root = ET.fromstring(content.decode("utf-8-sig", errors="replace"))
    except ET.ParseError:
        return [], None
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def _pt(el: ET.Element) -> dict[str, Any] | None:
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            return None
        ts_el = el.find(f"{ns}time")
        ele_el = el.find(f"{ns}ele")
        speed_el = el.find(f"{ns}speed")
        rec = {
            "ts": ts_el.text.strip() if ts_el is not None and ts_el.text else None,
            "latitude": lat,
            "longitude": lon,
            "heading_deg": el.get("course") or el.get("heading"),
            "altitude_m": ele_el.text.strip() if ele_el is not None and ele_el.text else None,
            "source": "gpx",
        }
        # GPX <speed> is m/s, not accuracy — recorded only as an issue note
        if speed_el is not None and speed_el.text:
            rec["_gpx_speed"] = speed_el.text.strip()
        return rec

    records: list[dict[str, Any]] = []
    for tag in ("trkpt", "rtept", "wpt"):
        for el in root.iter(f"{ns}{tag}"):
            rec = _pt(el)
            if rec:
                records.append(rec)
    return records, None


def parse_navigation_file(filename: str, content: bytes) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """Dispatch on extension.  Returns (raw_records, declared_crs, errors)."""
    name = (filename or "").lower()
    if name.endswith((".csv", ".txt")):
        records, crs = parse_csv(content)
        if not records:
            return [], crs, ["no usable lat/lon columns found in CSV"]
    elif name.endswith(".geojson") or name.endswith(".json"):
        records, crs = parse_geojson(content)
        if not records:
            return [], crs, ["no Point features found in GeoJSON"]
    elif name.endswith(".gpx"):
        records, crs = parse_gpx(content)
        if not records:
            return [], crs, ["no trkpt/rtept/wpt elements found in GPX"]
    else:
        # sniff: try GeoJSON then CSV
        if content.lstrip()[:1] in (b"{", b"["):
            records, crs = parse_geojson(content)
            if records:
                return records, crs, []
        records, crs = parse_csv(content)
        if not records:
            return [], crs, ["unsupported navigation file (use CSV, GeoJSON or GPX)"]
    return records, crs, []


# ---------------------------------------------------------------------------
# Public API: ingest + validate a whole file
# ---------------------------------------------------------------------------


def ingest_navigation(filename: str, content: bytes, *, file_crs: str | None = None) -> dict[str, Any]:
    """Full ingestion pipeline for one navigation file.

    Returns a diagnostics-first result::

        {
          "records": [...validated clean records...],   # valid only
          "stats":   {"total": n, "valid": v, "rejected": r,
                      "duplicate_timestamps": d, "impossible_jumps": j,
                      "gaps_over_60s": g},
          "rejected": [{"index": i, "issues": [...]}...],
          "warnings": [...],
          "crs":      declared/assumed CRS,
          "span":     {"start": iso, "end": iso} | None,
        }

    Rejected records are listed with reasons — never silently repaired,
    never stored as usable navigation (§3/§18).
    """
    raw_records, declared_crs, errors = parse_navigation_file(filename, content)
    if errors:
        return {
            "records": [],
            "stats": {
                "total": 0,
                "valid": 0,
                "rejected": 0,
                "duplicate_timestamps": 0,
                "impossible_jumps": 0,
                "gaps_over_60s": 0,
            },
            "rejected": [],
            "warnings": errors,
            "crs": declared_crs or file_crs,
            "span": None,
        }

    crs = file_crs or declared_crs
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []

    for i, rec in enumerate(raw_records):
        row_crs = rec.pop("_crs", None) or crs
        gpx_speed = rec.pop("_gpx_speed", None)
        clean, issues = _validate_record(rec, index=i, declared_crs=row_crs)
        if gpx_speed:
            warnings.append(f"record {i}: GPX <speed> ignored (not accuracy metadata)")
        if clean:
            ts = _parse_ts(clean["ts"])
            clean["_ts"] = ts
            valid.append(clean)
        else:
            rejected.append({"index": i, "issues": issues})

    # track-level checks (order/duplicates/jumps/gaps)
    stats_jumps = 0
    if valid:
        valid = _attach_track_diagnostics(valid)
        stats_jumps = sum(
            1 for r in valid if not r["valid"] and any("impossible jump" in x for x in r.get("issues", []))
        )
        # §3: track-invalidated records (jumps, duplicates) are surfaced in
        # the diagnostics WITH their reasons — never silently dropped.
        for r in valid:
            if not r["valid"]:
                rejected.append({"index": r.get("_index", -1), "issues": list(r.get("issues", []))})
        # gap census on the still-valid, dated records
        dated = sorted([r for r in valid if r.get("_ts")], key=lambda r: r["_ts"])
        gaps = sum(1 for i in range(1, len(dated)) if (dated[i]["_ts"] - dated[i - 1]["_ts"]) > GAP_THRESHOLD)
        dupes = sum(1 for r in valid if any("duplicate timestamp" in x for x in r.get("issues", [])))
        span = {"start": dated[0]["_ts"].isoformat(), "end": dated[-1]["_ts"].isoformat()} if dated else None
    else:
        gaps = dupes = 0
        span = None

    records = [r for r in valid if r["valid"]]
    return {
        "records": records,
        "stats": {
            "total": len(raw_records),
            "valid": len(records),
            "rejected": len(rejected),
            "duplicate_timestamps": dupes,
            "impossible_jumps": stats_jumps,
            "gaps_over_60s": gaps,
        },
        "rejected": rejected,
        "warnings": warnings,
        "crs": crs or "EPSG:4326",
        "span": span,
    }


# ---------------------------------------------------------------------------
# Timestamp synchronization (§4)
# ---------------------------------------------------------------------------


def sync_frame_to_track(
    *,
    ts: datetime | None,
    track: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Match one sonar timestamp to the navigation track.

    Method priority (§4): exact match → linear interpolation between the
    bracketing fixes → nearest (only when the frame time is outside the
    track span, where interpolation is impossible).  ``None`` when there is
    no usable track or no timestamp.  Never fabricates a position.
    """
    if not track or ts is None:
        return None
    dated = [r for r in track if r.get("_ts") is not None]
    if not dated:
        return None
    dated.sort(key=lambda r: r["_ts"])

    # exact match (±1 s window to absorb rounding noise)
    for r in dated:
        if abs((r["_ts"] - ts).total_seconds()) <= 1.0:
            return _sync_result(r, SYNC_EXACT, offset_s=(ts - r["_ts"]).total_seconds())

    # bracket → linear interpolation
    for i in range(1, len(dated)):
        a, b = dated[i - 1], dated[i]
        ta, tb = a["_ts"], b["_ts"]
        if ta <= ts <= tb:
            span = (tb - ta).total_seconds()
            if span <= 0:
                return _sync_result(a if (ts - ta).total_seconds() < (tb - ts).total_seconds() else b, SYNC_NEAREST)
            f = (ts - ta).total_seconds() / span
            lat = a["latitude"] + f * (b["latitude"] - a["latitude"])
            lon = a["longitude"] + f * (b["longitude"] - a["longitude"])
            hdg_a, hdg_b = a.get("heading_deg"), b.get("heading_deg")
            heading = None
            if hdg_a is not None and hdg_b is not None:
                # shortest-arc heading interpolation
                d = ((hdg_b - hdg_a + 180.0) % 360.0) - 180.0
                heading = (hdg_a + f * d) % 360.0
            alt_a, alt_b = a.get("altitude_m"), b.get("altitude_m")
            altitude = None if alt_a is None or alt_b is None else alt_a + f * (alt_b - alt_a)
            acc = [x for x in (a.get("accuracy_m"), b.get("accuracy_m")) if x is not None]
            return {
                "latitude": lat,
                "longitude": lon,
                "heading_deg": heading,
                "altitude_m": altitude,
                "accuracy_m": max(acc) if acc else None,
                "sync_method": SYNC_LINEAR,
                "nav_source": "uploaded_track_interpolated",
                "interpolation_fraction": round(f, 4),
                "bracket": [a.get("id"), b.get("id")],
                "gap_s": round(span, 3),
            }

    # outside the span → nearest endpoint, flagged as approximate sync
    before = ts < dated[0]["_ts"]
    ref = dated[0] if before else dated[-1]
    res = _sync_result(ref, SYNC_NEAREST, offset_s=(ts - ref["_ts"]).total_seconds())
    res["note"] = "frame timestamp outside navigation span — nearest fix used (not interpolated)"
    return res


def _sync_result(rec: dict[str, Any], method: str, *, offset_s: float = 0.0) -> dict[str, Any]:
    return {
        "latitude": rec["latitude"],
        "longitude": rec["longitude"],
        "heading_deg": rec.get("heading_deg"),
        "altitude_m": rec.get("altitude_m"),
        "accuracy_m": rec.get("accuracy_m"),
        "sync_method": method,
        "nav_source": "uploaded_track",
        "offset_s": round(offset_s, 3),
    }


# ---------------------------------------------------------------------------
# Towfish / layback (§7)
# ---------------------------------------------------------------------------


def towfish_position(
    *,
    vessel_lat: float,
    vessel_lon: float,
    heading_deg: float | None,
    offset_forward_m: float | None,
    offset_starboard_m: float | None,
) -> tuple[float, float] | None:
    """Vessel GNSS → towfish position via documented offsets (§7).

    ``offset_forward_m`` is along the heading, ``offset_starboard_m`` to
    starboard.  Returns None when heading or both offsets are missing —
    a missing layback is NEVER invented (§7).
    """
    if heading_deg is None:
        return None
    if offset_forward_m is None and offset_starboard_m is None:
        return None
    from .pipeline.geolocate import haversine_destination

    lat, lon = vessel_lat, vessel_lon
    if offset_forward_m:
        lat, lon = haversine_destination(lat, lon, heading_deg, offset_forward_m)
    if offset_starboard_m:
        lat, lon = haversine_destination(lat, lon, (heading_deg + 90.0) % 360.0, offset_starboard_m)
    return lat, lon


# ---------------------------------------------------------------------------
# Survey configuration (§7)
# ---------------------------------------------------------------------------

#: survey_meta keys consumed by the geolocation v2 path
SURVEY_CONFIG_KEYS = (
    "towfish_offset_forward_m",
    "towfish_offset_starboard_m",
    "layback_m",
    "sensor_depth_m",
    "antenna_offset_m",
    "heading_offset_deg",
)


def survey_towfish_config(meta: dict[str, Any]) -> dict[str, float | None]:
    """Read the documented towfish/layback config from survey metadata.

    Only values the operator supplied are returned; everything else stays
    None so downstream geolocation can honestly downgrade (§7).
    """
    return {k: _num(meta.get(k)) for k in SURVEY_CONFIG_KEYS}


# ---------------------------------------------------------------------------
# Misc small helpers used by the runner
# ---------------------------------------------------------------------------


def frame_timestamp(img: dict[str, Any]) -> datetime | None:
    """A frame's capture time (``captured_at`` column) as aware UTC."""
    return _parse_ts(img.get("captured_at"))
