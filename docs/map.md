# Survey Intelligence Map (MapLibre) — Phase 4

> **Plain-language explanation.** Think of the map as AQUA's underwater survey
> map. The sonar analysis finds possible objects. The database remembers them.
> The geolocation layer tells us where they can be placed when enough
> information exists. MapLibre lets the operator see those real results on a
> map. If AQUA doesn't know where something is, it leaves the location unknown
> instead of guessing.

The map is a **viewer**, never a source of truth. Every marker, track point and
count comes from the persisted database through the Phase 3 spatial API.

## A. Spatial API source

Primary data source (authoritative):

```
GET /api/surveys/{survey_id}/spatial
```

Returns the backend-computed survey summary (frame/target location counts),
geographic bounds, the observed survey **track**, and the **persistent target**
list with coordinates and provenance. The frontend performs **no geolocation
math** — it renders exactly what the backend reports.

Target history (fetched on selection only):

```
GET /api/surveys/{survey_id}/targets/{target_id}/history
```

The Phase 3 endpoint contract is treated as authoritative; Phase 4 introduced
**no backend changes** and no second spatial data source.

## B. Target marker semantics

- The map plots **persistent targets**, never raw frame detections.
- A marker exists **only** for a target whose backend-reported coordinates are
  valid (`latitude`/`longitude` non-null). A target with
  `geolocation_status: "unknown"` gets **no marker** — no fallback coordinates,
  no default city, no guessing.
- Each marker's accessible label carries target ID · class · status · location
  status (e.g. `Open target tgt_8566… · structure · active · location
  approximate`), so class/status/location are never color-only (§33).
- Markers are keyboard-focusable `<button>`s with descriptive `aria-label`s.
- The marker's position is the target's **representative location** — it does
  not imply object footprint, exact dimensions, or 3D position (§42).
- Marker styling reuses the shared `classColor` design tokens (wreck / debris /
  structure / tire); the selected target is visually distinguished.

## C. Track semantics

- The track is drawn **only** from `spatial.track` — real observed frame
  positions in frame order (`position_source: "observed"`).
- The backend never interpolates across missing positions; the frontend never
  invents points or draws segments through gaps.
- Track status wording matches the data: **Available** (`track.available`),
  **Incomplete** (real points exist but too few / not a full path),
  **Unavailable** (no observed positions). A single located frame yields
  `available: false` with a note — not a fabricated line.
- The track represents the **platform path**, not an object path.

## D. Filters

Client-side filtering of the authoritative dataset (server-side filtering is
unnecessary at prototype scale):

| Filter   | Values |
| -------- | ------ |
| Class    | all · wreck · debris · structure · tire |
| Status   | all · active · review · confirmed · rejected |
| Location | all · located · unavailable |

Filters update markers, the `Showing N marker(s) · M target(s) match filters`
count, and behave sanely with a selected target (stale selections clear; see
MapPage). Zero-match and match-but-unmappable are shown differently (§G).

## E. Location status handling

Wording preserves the Phase 3 semantics:

- Backend `approximate` → shown as **“Approximate location”**, source
  “frame navigation + sonar geometry”, with the backend uncertainty (e.g.
  `±9.6 m`) when provided. Never “exact”, never survey-grade.
- Backend `observed` → “Observed frame navigation”.
- Backend `unknown` → **“Location unavailable”** with the honest reason. Never
  “located”, never `0,0`.

Provenance is summarized in plain language (e.g. “estimated from frame
navigation + sonar geometry (never survey-grade)”) with an expandable details
section rather than raw JSON as primary UX. Target-level **location
consistency** (`consistent` / `inconsistent` / `insufficient_data`) is shown
exactly as the backend computed it, including spread in metres.

Confidence shown on the map/Lab is defined as *max evidence-fusion score of
associated detections* — a documented prototype aggregate, not a calibrated
probability.

## F. Map → Lab workflow

1. Open the Marine Intelligence Map and pick a survey (header shows the real
   survey name, frames, targets, located count, track status).
2. Survey track and located persistent targets render.
3. Filter by class / status / location.
4. Click a target marker → detail drawer (identity, class, status, location +
   provenance, observation count, first/last seen, location consistency,
   compact per-frame history with fusion scores).
5. **Open in Lab** navigates to `/lab/{surveyId}?target={targetId}`.
6. The Lab consumes the hand-off once targets load, expands that persistent
   target's row and clears the query param (`replace: true`). Unknown ids are
   ignored safely.

Map and Lab refer to the **same persisted target IDs** — there is no duplicate
target identity logic (§53). Reports use the same IDs (§54).

## G. No-location, empty and error states

- **No-GPS survey** (`frames_with_location = 0` and no located targets): a
  dedicated state explains that analysis succeeded but no valid navigation
  metadata existed to place results geographically. The map does not jump to an
  arbitrary location and no markers appear.
- **Partial GPS**: summary shows e.g. `Located: 2 / 5` and `Track:
  Incomplete`; unlocated targets simply have no marker. If the location filter
  is set to `unavailable` on such a survey, the empty state says
  “**N target(s) match, but none has a mappable location**” and points to the
  Lab — it never claims zero matches falsely (§27).
- **Zero matches after filtering**: “No targets match the current filters.”
  (shown over the track when the map is visible; as a full empty state when it
  is not).
- **Loading / errors**: explicit loading and error messages; 404/500 surface a
  clear “Unable to load spatial survey data” style message — no blank map, no
  silent failure. A selected target that disappears after a refresh clears the
  selection safely.
- **Refresh** re-fetches `/spatial` from the backend (no silent local
  mutation); **Fit survey** fits the real bounds when they exist.

## H. Known limitations

- **Deep links are not fixed.** The documented SPA-fallback issue remains:
  direct navigation to `/app/map?…` (or any `/app/…` deep link) returns the
  backend 404. Enter through `/app/` and navigate client-side (hash-based
  routing inside the app works). Fixing the server-side fallback is an
  infrastructure change outside Phase 4 scope.
- Map state (center/zoom) is not persisted across visits beyond the current
  session's component lifetime; only the selected survey and target flow
  through the URL.
- Markers are simple styled buttons, not MapLibre symbol layers; clustering is
  intentionally omitted (prototype-scale datasets — documented decision per
  §25).
- No heatmap, clustering, route optimization, telemetry, 3D, terrain,
  satellite layers or external GIS services (§58) — by design.
- Geolocation remains **approximate** wherever the backend says so; the map
  adds no precision the database does not claim.
