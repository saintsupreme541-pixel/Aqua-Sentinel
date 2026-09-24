# Metrics, evaluation conventions and the fallback baseline

## What we measure (and why not "accuracy")

Detection on sonar is class-imbalanced and spatially ambiguous; accuracy is
meaningless. The project reports, per operating point:

- **P / R / F1 @ IoU 0.5** (any-object level for the heuristic detector —
  it cannot claim per-class recognition)
- **false alarms per image** (survey tooling must not bury the analyst)
- **mean Dice** of highlight masks vs pixel ground truth (real-data subset)
- **ECE** (expected calibration error) for the fused confidence, when labels
  are available
- **human-review rate** (fraction of detections routed to an analyst)

Segmentation is reported as Dice/IoU, never box-only.

## Current baseline numbers (heuristic fallback, no weights)

Captured by `python backend/scripts/evaluate.py` (JSON in
`docs/metrics/baseline-heuristic.json`):

| Set | Images | GT objs | P | R | F1 | FA/img |
|---|---|---|---|---|---|---|
| Synthetic SSS survey | 7 | 7 | 0.556 | 0.714 | 0.625 | 0.571 |
| MD-FLS watertank (subset) | 24 | 52 | 0.220 | 0.519 | 0.309 | 4.000 |
| **Combined** | 31 | 59 | 0.242 | 0.542 | 0.335 | — |

Mask Dice on the real subset ≈ 0.41.

Read these as **an honest lower bound for the fallback**, not as product
performance: the baseline detector over-fires on cluttered FLS tank imagery
(walls, fan texture) and deliberately does not name classes. The product
differentiator — verification gating — is what converts detector candidates
into defensible confirmations; with ONNX weights registered, the same
harness re-runs against the trained model and the delta is the story.

## Calibration & operating point

Fusion uses a logistic opinion pool with per-signal weights renormalized over
available signals. Defaults (gain 3.2, intercept −0.4) place `confirmed` at
fusion ≥ 0.63 — a threshold that in practice requires **two or more
independent strong signals**, so a lone over-confident detector output stays
a `candidate`. Weights/intercepts are fit by `fusion.fit_fusion()` on
labelled folds for a real deployment, with ECE reported before/after.

## Rules the pipeline obeys

1. No fabricated coordinates — geolocation requires GPS+heading+altitude+side
   and carries a Monte-Carlo 2σ ellipse; otherwise `known=false`.
2. No claim that dark = acoustic shadow — shadow is optional evidence with
   geometric validity checks.
3. No silent "AI" — every result records its backend (`heuristic` vs `onnx`).
4. Synthetic data is labeled synthetic everywhere (manifest, UI, reports).
5. Splits are survey-aware; ping-level leakage is disclosed when it cannot be
   avoided (single-survey datasets).
6. Numbers reported on held-out splits only, with the split caveats.
