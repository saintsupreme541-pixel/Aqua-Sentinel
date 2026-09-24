# Geolocation Provenance — Phase 3

> **Human explanation.** Imagine a boat or underwater vehicle looking for an
> object. The vehicle knows where it is. The sonar sees something to the
> side. If we know the vehicle's position, direction and enough sonar
> information, we can *estimate* where that object is. AQUA remembers **how**
> it calculated that location. If we don't have enough information, AQUA
> says: "Location unavailable." It never makes up a location.

## A. Observed vs derived

| Kind | Fields | Where it comes from |
|---|---|---|
| **OBSERVED** | frame `latitude`, `longitude`, `heading_deg`, `altitude_m`, `sonar_side`, `slant_range_m` (sonar range), `captured_at` | operator / device / sonar metadata — stored NULL when absent |
| **DERIVED** | detection fix (`geolocation.lat/lon`), bearing, ground range, uncertainty ellipse, target representative location, location consistency, quality score | computed by AQUA from observed metadata + sonar geometry |

The distinction is machine-readable: every derived coordinate carries a
`provenance` object (`kind: "derived"`, `source`), and unavailable results
carry `kind: "unavailable"` with the list of missing/invalid inputs. Frame
positions in `/spatial` are labeled `position_source: "observed"`.

## B. Coordinate conventions

- WGS-84 geographic coordinates (lat/lon degrees), consistent with the rest
  of the stack (MapLibre `[lon, lat]` order in GeoJSON).
- Local scale: spherical-Earth haversine with `EARTH_RADIUS_M = 6,371,008.8 m`.
  Prototype-scale accuracy is fine; no local datum/UTM projection is used.

## C. Heading convention

- Degrees, **0–360 normalized** (`heading % 360`). 0° = north, 90° = east.
- Invalid inputs (NaN, infinite, non-numeric) are **rejected**, never
  silently wrapped: `normalize_heading` returns `None` and validation
  records `"heading missing or invalid"` in the provenance issues.

## D. Sonar-side convention

- `starboard` → object bearing = heading **+ 90°**
- `port` → object bearing = heading **− 90°** (mod 360)
- This is the abeam assumption of the existing AQUA geometry core
  (`geolocate.locate_detection`, unchanged since the original audit). Any
  dataset whose sensor does not look abeam must not use the derived fix.

## E. Slant-range interpretation

- `slant_range` = straight-line sensor→object distance.
- Ground range = `sqrt(slant² − altitude²)` on a **flat-seabed assumption**
  (`slant_to_ground`). Physically impossible pairs (slant < altitude) clamp
  to 0 and the fix is refused.
- Slant range is **never** reported as horizontal distance; the CSV carries
  separate `geo_slant/ground`-style columns and the provenance lists both.

## F. Approximate geolocation method (spec §12)

```
reference platform position (OBSERVED frame lat/lon)
+ heading (OBSERVED, normalized)
+ sonar side (OBSERVED)          → bearing = heading ± 90°
+ slant range (row → pixel_to_slant) + altitude (OBSERVED)
                                 → ground range = sqrt(sl² − alt²)
= approximate object position    → haversine_destination(...)
```

- Uncertainty: the pre-existing Monte Carlo over GPS/heading/range/altitude
  errors, summarized as a 2σ error ellipse (`uncertainty_m` = geometric
  mean radius).
- The result is **always labeled `approximate`** — never exact, never
  ground truth, **not** full 3D reconstruction. Frame position ≠ object
  position; the offset comes from sonar geometry.

## G. Statuses (spec §4)

| Status | Meaning |
|---|---|
| `approximate` | a derived fix exists (observed nav metadata + geometry). Still an estimate. |
| `unknown` | not enough *valid* metadata. Coordinates are null. **Unknown ≠ zero ≠ false ≠ low confidence.** |

(There is deliberately no "exact/available" status for derived fixes — only
observed frame positions carry `position_source: "observed"`.)

## H. Validation rules (spec §15)

`validate_nav` accepts only: lat ∈ [−90, 90], lon ∈ [−180, 180], finite
heading, finite altitude > 0, side ∈ {port, starboard}, finite range ≥ 0.
Invalid input ⇒ no coordinate is produced and the issue is named in
`provenance.issues`.

## I. Target aggregation method (spec §7/§8)

Phase 2 already selects a deterministic **representative detection** (max
evidence-fusion; ties → lowest frame/ids). Phase 3 keeps target location
*consistent with that rule*:

1. If the representative detection has a derived fix → **it is the target
   location** (representative observation rule).
2. Else → the located observation with the lowest `uncertainty_m` (ties →
   lowest ids) supplies it, and the provenance `selection_rule` records why.

Coordinates are **never averaged** (a mean of fixes across a moving track
can fall on unusable seabed) and never fabricated. The chosen observation
is recorded in `geolocation_evidence.from_detection_id` / `from_frame_id`.
Aggregation runs inside the association transaction and is **idempotent**
(§36): recomputation is a pure function of the linked detections, so a
second run yields byte-identical state.

## J. Multi-frame location consistency (spec §17–§18)

Distinct from *association* ("same object?"), consistency asks: "do the
target's located observations agree?"

- `dispersion_m` = max pairwise haversine distance between the fixes.
- Threshold = `max(2 × mean uncertainty, 20 m)` — a documented heuristic
  (`LOCATION_CONSISTENCY_BASE_M`), **not** a calibrated constant.
- `status`: `consistent` / `inconsistent` / `insufficient_data` (< 2
  located observations — honest, not zero).

## K. Unavailable-data behavior

- Frame nav columns stay NULL — never fabricated retroactively.
- Detections: `geolocation.status = "unknown"`, lat/lon null, provenance
  `kind: "unavailable"` with `issues` naming exactly what's missing.
- Targets: `geolocation_status: "unknown"`, null coordinates,
  `location_note` explains "no observation had enough valid metadata".
- Survey track: contains **only observed frame positions**, in frame order;
  gaps are **not** interpolated; fewer than 2 points ⇒ `track.available =
  false`. No positions ⇒ empty coordinates, null bounds.

## L. Precision limitations

- Stored coordinates keep full float precision; the **UI displays 4 decimal
  places (~11 m)** and the PDF shows 5 (≈ 1.1 m digit) with the word
  "approximate" — display never implies survey-grade accuracy.
- The Monte Carlo uncertainty uses assumed instrument errors (2° heading,
  5 m GPS, 5% range/altitude) — indicative, not calibrated per sensor.

## M. AUV / USV considerations

- **USV:** GNSS + heading usually available per frame → derived fixes
  straightforward.
- **AUV submerged:** GPS generally unavailable; INS/DVL estimates may be
  supplied per frame by the operator but are the operator's provenance to
  declare. AQUA never invents them.
- Both paths use the same nullable schema — the architecture is
  vehicle-agnostic and degrades to `unknown` honestly.

## N. Known limitations

1. Flat-seabed, abeam-only geometry; sloped seabed or non-abeam sensors
   invalidate the derived fix.
2. Single-frame surveys give per-detection fixes only; consistency needs
   ≥ 2 located observations.
3. The uncertainty model's error parameters are prototype assumptions.
4. No map clustering/track UI yet (Phase 4); the Map page consumes the
   authoritative `/spatial` data but remains the pre-existing layout.
5. A metadata-rich real survey is still required for a full live
   geolocation E2E (see the Phase 3 report) — the math itself is proven by
   deterministic unit tests against hand-computed coordinates.
