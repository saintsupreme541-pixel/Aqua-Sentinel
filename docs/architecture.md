# AQUA-SENTINEL — Architecture

## 1. Pipeline (per survey)

```
Survey upload (images + metadata)
  │  raw bytes preserved unchanged (storage/raw/)
  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ per-image stages (worker thread, SSE progress)                          │
│                                                                         │
│  quality assessment → preprocessing (light/standard/aggressive)         │
│    → detection → per-candidate: segmentation · classify · shadow        │
│      · anomaly → geolocation · dimensions                               │
│  artifacts: processed.png · mask_*.png · overlay.png (all stored)       │
└─────────────────────────────────────────────────────────────────────────┘
  ▼
survey-level pass
  consistency (multi-frame, within distance_m)
  → calibrated fusion (logistic opinion pool + per-signal breakdown)
  → verdict (confirmed / review / candidate / human_review_required)
  → priority (0–100, explainable factors)
  ▼
Reports: CSV · JSON · GeoJSON · PDF        Dashboard / Map / Explorer / Lab
```

Raw inputs are never overwritten: every derived artifact references its
preprocessing parameters, so any result is reproducible from the raw file.

## 2. Evidence signals and fusion

| Signal | Meaning | Availability gate |
|---|---|---|
| `detection` | detector score (heuristic salience or YOLO conf) | always |
| `segmentation` | box↔mask agreement, bell-shaped (weak if mask swallows box) | mask computed |
| `natural` | P(artificial) from classifier (uniformity, geometry, filament) | always |
| `shadow` | acoustic-shadow physics evidence | only when geometry known |
| `physics` | sonar-geometry plausibility (grazing window, height estimate) | only when altitude+range metadata present AND shadow valid |
| `consistency` | same object seen on adjacent pings within `distance_m` | survey with ≥2 images |

Fusion (see `backend/app/pipeline/fusion.py`):

```
logit = b0 + Σ (w_i / Σw_active) · p_i        conf = σ(gain · logit)
```

Weights are renormalized over **available** signals only — absence of
evidence is not evidence of absence. `fit_fusion()` fits weights by gradient
descent on (signals → label) pairs; `expected_calibration_error()` is
reported in metrics. The default operating point (gain 3.2, intercept −0.4,
confirm ≥ 0.63) is chosen so that confirmations require **at least two
independent strong signals**, which suppresses single-detector false alarms.

The default weights are initial *prototype* values, documented as tunable
(`detection 0.30 · segmentation 0.15 · natural 0.20 · shadow 0.15 ·
physics 0.10 · consistency 0.10`; env-overridable gain/intercept in
`backend/app/config.py`) and replaced wholesale by `fit_fusion()` when
labelled data exists.

Physics-informed analysis (`app/pipeline/physics.py`): on a flat-seabed
assumption the shadow/height relation is a similar-triangles estimate with
a grazing angle `θ = asin(altitude / slant_range)` computed per detection
row. Shadows are only well formed within a grazing window (8°–65°), so the
geometry score decays outside it and the note tells the user to treat the
height with caution. **No 3D reconstruction is claimed** — when altitude or
range metadata is missing, `metadata_sufficient=false` and the physical
quantities are `null` while image-based shadow evidence remains valid.

Anomaly is deliberately **not** a fusion input: a rock fits the background
model well, so rewarding "low anomaly" would confirm natural objects. It acts
only as an override: `anomaly ≥ 0.8` and `class_conf < 0.6` ⇒
`human_review_required`.

## 3. Fallback heuristics (why they exist, and their limits)

To make the entire product runnable offline, three deterministic baselines
are implemented. They are clearly labeled as such in the UI, reports and
metrics, and are replaced automatically when ONNX weights are registered.

- **Detector** (`app/fallback/candidates.py`): Difference-of-Gaussians blob
  cue **and** per-row robust CFAR confirmation, then connected components
  filtered by size, shape, seed density and solidity. Emits conservative
  untyped `debris` (0.35–0.65) — it does **not** claim per-class recognition.
- **Segmenter** (`app/fallback/segment_threshold.py`): threshold from a ring
  around the box (local seabed statistics), largest-component cleanup.
- **Classifier** (`app/fallback/classifier_heuristic.py`): intensity
  uniformity, edge density, geometry and sparse-bright filament cues ⇒
  P(artificial). Filament texture is recorded as *evidence only* (semantic
  review v3: no supervised net class exists, so the class label is never
  rewritten to net_like/ghost_net — such signatures surface via the
  Unknown-Anomaly pathway).

Honest limits: the baseline detector over-fires on cluttered FLS tank imagery
(false alarms per image ≈ 4 on the watertank subset) and cannot name the
object class. A trained YOLO/U-Net is expected to raise precision markedly;
that comparison is the point of `scripts/evaluate.py`.

## 4. Geolocation & uncertainty

`app/pipeline/geolocate.py` implements flat-seabed slant-range geometry:

```
ground_range = √(slant² − altitude²)
bearing      = heading ± 90° (starboard/port)
fix          = haversine destination from vessel GPS
```

A Monte Carlo over GPS (5 m), heading (2°), range (5%) and altitude (5%)
errors yields a 2σ uncertainty ellipse rendered on the map and exported to
GeoJSON. If GPS, heading, altitude or side is missing the result is
`known=false` **with a null coordinate** — never a fabricated fix.

## 5. Shadow physics

`app/pipeline/shadow.py` looks for a dark run adjacent to the detection
highlight in the illumination (across-track) direction, with geometric
validity checks (direction of arrival, slant-range geometry, sane extent).
When valid:

```
object height ≈ altitude · shadow_ground_length / far_ground_range
```

Shadow length in *ground* range is derived from slant-range rows, so height
estimates are geometry-correct, not pixel heuristics. Not every object shows
a shadow (geometry, elevation, orientation) — shadow is optional evidence.

## 6. Storage

SQLite (single writer, WAL) at `data/runtime/aqua.db`; artifact files under
`data/runtime/{raw,processed,masks,overlays,reports}/`. PostGIS is a
drop-in swap for production (same data model), per the design docs.

## 7. Frontend

Vite + React + Tailwind, 7 routes: Dashboard · Upload · Sonar Intelligence
Lab · Detection Explorer · Marine Intelligence Map (SVG, offline-capable) ·
Hazard Intelligence · Reports. Types mirror the backend schemas exactly
(`frontend/src/api.ts` ↔ `backend/app/schemas.py`).
