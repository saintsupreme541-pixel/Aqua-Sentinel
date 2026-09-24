# SIH demo script — AQUA-SENTINEL AI

Total ≈ 8 minutes. Every step is offline and needs no model weights.

## Setup (before the demo, 2 terminals)

```bash
# terminal 1 — backend
cd backend && .venv\Scripts\activate        # or source .venv/bin/activate
python scripts/prepare_sample_data.py       # once — bundles real-data subset
python scripts/generate_synthetic_sss.py    # once — renders synthetic surveys
uvicorn app.main:app --port 8000            # http://127.0.0.1:8000/app

# terminal 2 — frontend (only if not using the single-port build)
cd frontend && npm run dev                  # http://127.0.0.1:5173
```

Suggested single-port mode for the demo: `cd frontend && npm run build`
beforehand; the backend then serves the dashboard itself at
`http://127.0.0.1:8000/app`.

## The narrative arc

### 1. The problem (30 s)
Read the framing: "Sonar experts eyeball thousands of images; rocks and sand
ripples look like pipes; single-model AI cries wolf." State the differentiator
in one sentence: *propose with a detector, verify with physics and
multi-model evidence, geolocate, prioritize.*

### 2. Upload Survey page → Import the **curated real-data demo** (90 s)
Click **Import** on *MD-FLS Watertank (curated demo)*. Call out that raw
files are kept untouched, preprocessing records every step, and the SSE
progress stream drives the UI. When done, open in the Lab.

### 3. Sonar Intelligence Lab — the multi-stage story (3 min)
Pick an image with a confirmed detection:
- **Detections tab** — overlay with boxes. Open a detection: show the
  **evidence fusion breakdown** (detector, segmentation agreement,
  artificial-probability, shadow, physics, consistency). Say: "No single
  number — the fused confidence is explainable; weights renormalize when
  shadow, physics or multi-frame evidence is unavailable — absence is not
  counted against the object."
- Toggle **Original vs Processed** — explain the *light* preset
  (aggressive denoising destroys small targets) and that preprocessing is
  reproducible from the raw file.
- If an image shows only candidates: "the fallback detector over-fires; the
  verification stack demotes false alarms to candidates instead of screaming
  'net!'."

### 4. Synthetic SSS survey → Map + Hazard pages (2 min)
Import **Synthetic SSS survey (GPS demo)**.
- **Marine Intelligence Map (MapLibre):** vessel track line, tier-colored
  hazard markers, click-to-inspect cards with coordinates, 2σ uncertainty
  and height estimates. Emphasize: *estimated* positions, slant-range
  corrected from GPS/heading/altitude; missing metadata ⇒ the system
  refuses to fabricate a fix ("Location unavailable").
- **Hazard Intelligence:** ranked cleanup priorities with expandable
  "why this score" factor bars. Note (post semantic review v3): no net
  class is claimed — the highest-ranked verified class leads; objects with
  anomalous signatures route to Human Review instead.

### 5. Reports (1 min)
Generate **GeoJSON + PDF** for the synthetic survey. Open the PDF (field
summary); note GeoJSON carries uncertainty polygons for GIS, and each
detection records its backend (honest about heuristic-vs-model).

### 6. The honest close (30 s)
"Baseline metrics are in docs/metrics.md — modest, as a no-weights fallback
must be. Drop in ONNX weights via the registry and re-run evaluate.py; the
product then shows a real model's numbers, measured the same way." Mention
calibration (ECE), survey-aware splits, synthetic-data labeling, and the
human-review override for anomalies. Never overclaim.

## Judge Q&A cheat sheet

| If asked | Answer |
|---|---|
| Rocks look like pipes — how do you stop false alarms? | Multi-stage verification: segmentation agreement, natural-vs-artificial, shadow physics (optional), multi-frame consistency, calibrated fusion; operating point requires two+ independent signals to confirm. |
| What accuracy do you get? | We report P/R/F1, Dice, false alarms/image, ECE — and the honest fallback numbers. Accuracy is meaningless here. |
| Did you train on real nets? | No public SSS net dataset exists — we scope *net-like filamentous debris*, trained partly on labeled synthesis, clearly labeled. |
| Are the coordinates real? | Synthetic survey coords are labeled synthetic. Real surveys: estimated fix + 2σ ellipse from metadata; never fabricated. |
| Is every dark area a shadow? | No — shadow requires highlight adjacency in the illumination direction and geometric sanity; it's optional evidence. |
| Heuristic vs AI? | The registry auto-uses ONNX weights when present; UI/reports state which backend ran. This demo runs the honest baseline. |
