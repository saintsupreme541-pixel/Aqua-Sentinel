# Multi-Frame Association — Persistent Targets (Phase 2)

> **Analogy.** Imagine taking four pictures of the same car while walking past
> it. The computer shouldn't say "four pictures means four cars." It should
> compare the observations and say "these pictures probably show the same
> car." That is what the persistent-target system does for sonar detections.

## What it is — and what it is NOT

Phase 2 adds a **deterministic, explainable association engine** that groups
frame detections across a survey's frames into *persistent targets* — one
database entity per likely physical object.

It is:

- deterministic (same inputs → same associations; no RNG)
- explainable (every decision stores per-signal scores + human-readable reasons)
- survey-scoped (a target in Survey A can never receive a detection from Survey B)
- idempotent (running twice never duplicates targets)
- transactional (one SQLite transaction per run; failures roll back cleanly)

It is **not**:

- a learned tracker (no neural networks, no ByteTrack/DeepSORT/SORT embeddings)
- a scientifically solved identity problem — see [Limitations](#limitations)
- a source of fabricated data (unavailable evidence stays unavailable)

## Pipeline position

```
Frame → AI analysis (unchanged ONNX pipeline) → frame detections persisted
      → survey-level association (Phase 2, after fusion)      → persistent targets
```

`runner.py` runs `targets.run_association(survey_id)` as the final survey
pass (stage `associate`, 99%). It can also be triggered manually:
`POST /api/surveys/{id}/associate` (idempotent) or with
`{"reset": true}` to explicitly rebuild after re-analysis.

## Signals (in the order the specification prefers)

| Signal | Source | When available | Weight |
|---|---|---|---|
| `class` | detection class vs target canonical class | always | 0.35 |
| `geographic` | haversine distance between geolocations | both sides genuinely located | 0.30 |
| `spatial` | IoU of bounding boxes | frames share the same dimensions | 0.20 |
| `frame_proximity` | `abs(frame_index)` gap | both frames have an index | 0.15 |

- **Unavailable ≠ zero.** Missing signals are excluded and the remaining
  weights renormalise; the decision records which signals were unavailable.
- **Conflicts are strong negatives applied after renormalisation:**
  class mismatch ×0.35, geographically-far-but-located ×0.35, disjoint
  boxes in comparable adjacent frames ×0.50. A wrong merge destroys
  information; a missed merge just leaves two targets an operator can merge.

## Score & thresholds (prototype heuristics — documented, not validated)

`score = Σ(weight·signal) / Σ(weight)` over available signals, then conflict
penalties. All values live in `app/association.AssociationConfig`:

- `accept_threshold = 0.62` — at/above: associate
- `ambiguity_margin = 0.06` — top two candidates within this margin →
  **ambiguous**, detection stays unassociated (never merged arbitrarily)
- geographic window: full score ≤ 8 m, zero ≥ 40 m (linear between)
- frame window: full score at gap ≤ 1, zero at gap ≥ 8
- spatial window: full at IoU ≥ 0.55, zero at IoU ≤ 0.05

## Deterministic target rules

- **Processing order:** `frame_index` ascending; stable `(created_at, id)`
  fallback. Never random DB order.
- **Representative detection:** highest evidence-fusion; ties break on
  lowest `(frame_index, image id, detection id)`.
- **first/last seen:** min/max associated frame — actual observations,
  never target-creation time.
- **Target confidence:** *max fusion of associated detections* — a
  documented aggregate, NOT a calibrated probability. Single-observation
  targets stay `active` (never auto-`confirmed`).
- **Target location:** representative detection's geolocation when known,
  else the linked detection with the lowest uncertainty; status
  `approximate` or `unknown`. Never fabricated.
- **Image-space signal** is only used when *all* frames in the survey share
  the same processed dimensions (a single comparable coordinate space).

## API surface (extends Phase 1, no duplicates)

- `GET  /api/surveys/{id}/targets` — list with observation counts
- `GET  /api/surveys/{id}/targets/{tid}/history` — identity + per-frame
  detection history in deterministic frame order, with the stored
  association evidence for each observation
- `POST /api/surveys/{id}/associate` — run the engine; returns the summary
  `{frames_processed, detections_processed, targets_created,
  detections_associated, already_associated, ambiguous, ...}`
- Phase 1 routes (create/patch/delete/associate-single) unchanged.

## Frontend

The Sonar Intelligence Lab gained a **Persistent targets** panel: lists the
survey's real targets (class, status, frames observed, confidence,
location status), an "Associate frames" button (calls the API; shows the
real summary), and an expandable per-target history view. Everything shown
comes from the backend — no fabricated cards.

## Limitations

1. **Heuristic, not learned.** Weights/windows are engineering choices for
   the prototype, documented as such. They are not fitted to ground truth.
2. **Metadata-dependent.** Without geolocation, association relies on frame
   proximity + box overlap in a shared coordinate space — weaker evidence;
   the engine stays conservative and prefers separation.
3. **No appearance model.** Visual similarity was deliberately excluded
   (Phase 2 forbids embedding/re-ID networks); two visually similar objects
   close together can be merged — ambiguity handling mitigates but does not
   eliminate this.
4. **`frame_index` is order, not time.** Surveys without indices degrade to
   no frame-proximity evidence.
5. **No target splitting/merging UI yet.** Wrong merges can be undone via
   `Rebuild` (reset) or Phase 1's unlink endpoint; operator tooling is a
   later phase.
6. **Non-real-time.** Association runs per-survey (batch), not per-frame
   streaming.

## Real-data status

The repository's real sonar assets (MD-FLS watertank) are independent tank
scenes, **not** a temporal sequence of one object, so the live E2E verifies
the machinery end-to-end on real ONNX output (11 detections → 5 targets,
all linked, idempotent) rather than multi-frame identity. Temporal-sequence
behavior is covered by the 12 synthetic cases in
`backend/tests/test_association_phase2.py`. No runtime data is fabricated.
