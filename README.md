# AQUA-SENTINEL

**AI-Powered Underwater Marine Intelligence using Side-Scan Sonar**

AQUA-SENTINEL analyzes side-scan sonar imagery to identify potential man-made underwater objects, evaluate each candidate with multiple independent evidence signals, persist observations across a survey into stable targets, estimate object location when sufficient navigation metadata exists, and give a human operator a map, evidence Lab, and reporting workflow.

Built for **Smart India Hackathon 2026, Problem Statement 26057** — *"AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery."*

---

## In one simple sentence

> **AQUA-SENTINEL is like an underwater detective: it detects, checks, remembers, locates, and helps a human operator review what it found.**

---

## The problem it solves

Inspecting the underwater environment is hard:

- Humans cannot see far underwater; **side-scan sonar** produces acoustic imagery instead of photographs.
- Manual review of sonar imagery is slow, and **natural seabed features (rocks, ripples) can look like man-made objects**.
- A single detector pass produces false confidence — the same object seen in several survey frames must be **recognized as one object, not many** (an identity problem).
- Knowing *where* an object is depends on the **navigation and sonar metadata available at capture time** — which is often missing.

AQUA-SENTINEL addresses each of these directly: propose with a detector, verify with independent evidence, persist across frames, locate honestly, and keep the human in charge of decisions.

## What is side-scan sonar?

> *A camera uses light; sonar uses sound.*

Side-scan sonar emits acoustic pulses and builds an image from the returning echoes. Bright regions reflect sound strongly; dark "shadow" regions behind tall objects absorb it. **This is an acoustic image, not a conventional optical photograph** — which is exactly why generic optical-vision pipelines struggle and why sonar-specific evidence (shadow geometry, across-track range physics) matters here.

---

## How it works — the pipeline

```text
Side-Scan Sonar Input
        ↓
Quality Assessment
        ↓
Preprocessing
        ↓
YOLOv8 Detection (ONNX)
        ↓
U-Net Segmentation (ONNX)
        ↓
Artificial / Natural Classification (ONNX)
        ↓
Acoustic Shadow Analysis
        ↓
Physics-Informed Shadow Geometry
        ↓
Unknown-Anomaly Scoring
        ↓
Multi-Frame Consistency
        ↓
Evidence Fusion → Status Decision
        ↓
Cleanup-Priority Scoring
        ↓
Persistence (SQLite)                    ← Phase 1
        ↓
Multi-Frame Association → Persistent Targets   ← Phase 2
        ↓
Geolocation + Provenance                ← Phase 3
        ↓
Survey Spatial Intelligence (authoritative /spatial API)
        ↓
MapLibre Map / Evidence Lab / Reports   ← Phase 4 + UI
```

Every detection carries a **fusion breakdown** showing which evidence signals pushed its confidence up or down — nothing is a single unexplained number.

### The core principle: propose → verify

On sonar imagery a lone YOLO box cannot defend "this is debris." So AQUA-SENTINEL **proposes** with a detector and then **verifies** each candidate with independent signals:

| Signal | What it checks | When absent |
|---|---|---|
| Detection | YOLO class confidence | — |
| Segmentation | U-Net mask agreement with the box (bell-shaped: too little or too much mask coverage both reduce trust) | scores 0.5 |
| Natural/artificial | CNN probability the object is man-made | — |
| Shadow | Dark region adjacent to the highlight, in the illumination direction; gives a physics-based **object-height estimate** | excluded (absence of shadow is never counted against an object) |
| Physics | Slant-range shadow geometry — only scored when the metadata makes it estimable | excluded and weights renormalized |
| Anomaly | Feature-space "unknown" score — high anomaly + low confidence routes to **human review** (an override flag, never positive evidence) | heuristic fallback |
| Consistency | Does this object look the same across overlapping frames? | excluded |
| Geolocation | Where is it? Only when navigation metadata allows | `unknown` — never fabricated |

Unavailable evidence is **excluded with renormalized weights, never scored as zero** — the system is vehicle-agnostic and works with whatever metadata the operator has.

---

## Core features (what is actually implemented)

### 1. Sonar image analysis
Quality assessment (flags unusable imagery, reports metadata completeness), light-touch preprocessing (normalization; optional despeckle/CLAHE — aggressive denoising destroys small targets), then the full evidence pipeline per frame.

### 2. YOLOv8 object detection (ONNX)
Unified YOLOv8n detector over **four trained classes only** (below). Operates on 640×640 letterboxed input with boxes mapped back to original image coordinates.

### 3. U-Net segmentation (ONNX)
Per-candidate masks (512×512 adapter, full-res logits mapped back out), stored as overlayable PNGs.

### 4. Artificial-vs-natural classification (ONNX)
Small CNN (64×64) scoring how likely the object is man-made; a sparse-bright filament texture may raise this *as evidence only* — class labels are never rewritten.

### 5. Acoustic-shadow + physics analysis
Shadow presence, extent, and a slant-range-corrected object-height estimate. Slant range is **never** confused with horizontal distance: `ground = √(slant² − altitude²)`, and physically invalid geometry (`slant ≤ altitude`) is rejected, yielding `status: unknown`.

### 6. Evidence fusion & transparent status
Logistic opinion pool with documented, tunable weights (env-tunable; defaults in `backend/.env.example`, rationale in `docs/metrics.md`). Produces a fused confidence **plus a per-signal breakdown**, and decides one of: `confirmed`, `review`, `candidate`, `human_review_required` — the latter when a high anomaly score meets a low class confidence.

### 7. Cleanup-priority scoring
Transparent 0–100 weighted score from class hazard, fused confidence, and (when estimable) physical dimensions — binned into priority tiers. The dashboard's High/Medium/Low view maps to these **real tiers**; no separate risk model is invented.

### 8. Persistent targets across frames
Deterministic, explainable, survey-scoped multi-frame association (no learned tracker): class compatibility, frame-sequence proximity, geographic proximity where positions exist, image-space IoU between comparable frames — with documented thresholds, an ambiguity margin, and an anti-over-merge guard. Every accept/reject decision is stored and inspectable. Same inputs → same targets, every time.

### 9. Geolocation with provenance
When the operator supplies GPS/heading/altitude/range and the sonar side: object bearing = `heading ± 90°` (starboard +, port −), ground range from slant range, position by haversine destination, with a Monte-Carlo **2σ uncertainty ellipse**. Every fix records **observed vs derived** inputs, which frame provided the reference, which assumptions were used, and why. **Never fabricates coordinates** — missing metadata yields `unknown`, not 0,0.

### 10. Survey spatial intelligence
The authoritative `GET /api/surveys/{id}/spatial` endpoint serves frame-position tracks (observed points only — gaps are never interpolated), per-target locations, bounds, and availability counts. The frontend renders exactly what this returns; it never recomputes geography.

### 11. Marine Intelligence Map (MapLibre)
Survey track + persistent-target markers with class, status, and location status; class/status/location filters; a target detail drawer (provenance, consistency, history); and **Open in Lab** hand-off. Targets without valid coordinates get **no marker** — an honest "Location unavailable" state instead.

### 12. Sonar Intelligence Lab
Frame-by-frame evidence inspection: quality, preprocessing parameters, detections with overlays, segmentation masks, shadow/physics/anomaly/consistency/fusion breakdowns, geolocation provenance, and target history — every number traceable.

### 13. Reports
Per-survey **CSV, JSON, GeoJSON, and PDF** reports generated from the same authoritative database as the API — including geolocation provenance and uncertainty polygons (GeoJSON uses `[longitude, latitude]` order; standard API objects use `latitude`/`longitude`).

### 14. Operator workflow & animated dashboard
Upload Survey → SSE-streamed analysis progress → Lab / Map / Hazard / Reports. The dark deep-ocean interface (custom animation system, ambient background, reduced-motion support) is documented in [`docs/frontend-visual-system.md`](docs/frontend-visual-system.md).

### 15. Anomaly / human-review pathway
A feature-space unknown-score exists so that *untrained-but-suspicious* objects (e.g. filamentous net-like textures) surface as `human_review_required` rather than being forced into a trained class that has no verified data behind it.

---

## Detection taxonomy

The detector is trained on **exactly these four classes**:

| ID | Class |
|----|-------|
| 0 | `wreck` |
| 1 | `debris` |
| 2 | `structure` |
| 3 | `tire` |

> The taxonomy is intentionally limited to classes supported by the current training data. Nothing outside it is ever emitted; untrained-but-suspicious objects route through the anomaly/human-review pathway instead of being mislabeled.

---

## System architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Frontend — React 18 + Vite + Tailwind (dark marine command center)  │
│  Dashboard · Upload · Lab · Explorer · Map (MapLibre) · Hazards ·     │
│  Reports                                                             │
└───────────────▲──────────────────────────────────────────────────────┘
                │ typed REST + SSE (server-sent events for job progress)
┌───────────────┴──────────────────────────────────────────────────────┐
│  Backend — FastAPI (Python 3.11+)                                    │
│                                                                      │
│  API layer: surveys · frames · detections · targets · spatial ·      │
│             jobs (SSE) · reports · media · samples · dashboard ·     │
│             /api/analyze (stateless single-image)                    │
│                                                                      │
│  Pipeline: quality → preprocess → detect → segment → classify →      │
│            shadow → physics → anomaly → geolocate → dimensions →     │
│            consistency → fusion → priority → association             │
│                                                                      │
│  Model registry (models/registry.json): ONNX backends with           │
│  deterministic heuristic fallback — the active backend is labeled    │
│  on every result, never silently substituted.                        │
└───────────────▲──────────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────────────────────────────────────────────┐
│  SQLite (single file, WAL) + artifact store                          │
│  surveys → frames (images) → detections → targets (+ jobs, reports)  │
│  raw/ processed/ masks/ overlays/ reports/                           │
└──────────────────────────────────────────────────────────────────────┘
```

**The database is the source of truth.** After a restart, every survey, frame, detection, target, and artifact reference is recoverable — verified by regression tests.

---

## What the user sees

A dark, marine-intelligence command center:

1. **Dashboard** — real summary cards (surveys, frames, detections, priority tiers), a sonar analysis panel with the actual selected frame and its real evidence overlays, an embedded map, searchable detected-object list, and quick report actions.
2. **Upload Survey** — drag-drop sonar frames, optional per-frame navigation metadata (lat/lon, heading, altitude, side, range), importable demo samples, live SSE progress.
3. **Sonar Intelligence Lab** — pick a survey/frame and inspect every stage's real output side by side.
4. **Detection Explorer** — filter the full detection table.
5. **Marine Intelligence Map** — survey track, target markers, filters, detail drawer, Open-in-Lab.
6. **Hazard Intelligence** — priority-tier review workflow.
7. **Reports** — generate and download CSV / JSON / GeoJSON / PDF.

---

## Technology stack

| Layer | Technology |
|---|---|
| Detection / Segmentation / Classification | YOLOv8n + U-Net + CNN, served via **ONNX Runtime** (`onnxruntime`) |
| Backend | Python 3.11+, FastAPI, Pydantic v2, Uvicorn |
| Imagery | OpenCV (headless), NumPy, Pillow |
| Reports | ReportLab (PDF) + CSV/JSON/GeoJSON |
| Database | SQLite (WAL mode), single-file persistence |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, MapLibre GL, lucide-react, React Router |
| Real-time progress | Server-Sent Events (SSE) |
| Packaging | pip (`pip install -e ".[dev]"`), Docker Compose, Makefile |

---

## What enters the system (data inputs)

- **Sonar frames** — PNG/JPG/TIFF/BMP/WEBP side-scan (or forward-looking) sonar images, one survey per batch.
- **Optional navigation metadata** — latitude, longitude, heading (deg, 0–360), altitude (m), sonar side (`port`/`starboard`), slant/maximum range (m), capture timestamp. Supplied per-survey as defaults and/or per-frame overrides (**frame metadata always wins** — survey values are fallback only).
- **Demo samples** — `data/samples/` bundles an MD-FLS watertank subset (real data, **no GPS**) and generated synthetic SSS surveys (with synthetic nav metadata, clearly labeled).

Nothing else: the system fabricates no GPS, no heading, no detections, no timestamps. Missing metadata simply flows through as "unavailable."

## What the system never does

- ❌ Never fabricates coordinates, headings, detections, confidence values, or timestamps
- ❌ Never presents a derived position as exact (derived fixes are always labeled `approximate`)
- ❌ Never silently substitutes a missing metadata value with zero or a guess
- ❌ Never counts absent evidence as negative evidence
- ❌ Never emits classes outside the four-class taxonomy
- ❌ Never lets the frontend compute geography — the backend spatial API is the only source

---

## Running the project

Requirements: **Python ≥ 3.11**, **Node ≥ 18**.

### Local development

```bash
# 1. Backend
cd backend
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                            # http://127.0.0.1:5173 (proxies /api → :8000)

# 3. Backend server (new terminal)
cd backend
.venv/Scripts/python -m uvicorn app.main:app --port 8000    # http://127.0.0.1:8000/docs
```

Open the dashboard → **Upload Survey** → import a demo sample (or upload your own frames) → watch the live progress → explore in Lab / Map / Reports.

### Single-port demo (production-style)

```bash
cd frontend && npm run build           # → frontend/dist
cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000
# Dashboard is now served by the backend at http://127.0.0.1:8000/app
```

Verified Windows quick start (PowerShell, from the repo root):

```powershell
cd frontend; npm run build; cd ..
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# → http://127.0.0.1:8000/app  (dashboard)  ·  /docs  (API)
```

Health check: `curl http://127.0.0.1:8000/api/health` should report `"mode":"full-model"` with all three ONNX backends loaded.

### Docker

```bash
docker compose up --build              # backend :8000, frontend :5173 (nginx proxies /api)
```

### Makefile shortcuts

```bash
make setup        # venv + backend deps
make samples      # prepare demo data
make test         # backend pytest suite
make lint         # ruff + eslint
make typecheck    # mypy + tsc
make build        # frontend bundle
make demo         # build frontend + single-port backend on :8000
```

### Configuration

All settings are environment variables with sensible defaults (see `backend/.env.example`): storage/DB paths, upload limits, fusion weights and thresholds, uncertainty scale, and the model registry path.

### Model weights

`models/registry.json` declares which backend serves each task. Bundled ONNX weights (in `backend/app/models/` / `models/weights/`) run offline with no GPU; if weights are missing, the documented heuristic fallback activates and the API/UI **clearly labels** the degraded mode (`mode: degraded-heuristic` in `/api/health`) — trained and heuristic results are never silently conflated.

---

## Validation status

All gates green on the current tree:

| Check | Result |
|---|---|
| Backend tests (`pytest`) | **200 passed**, 1 skipped (17 test files: pipeline, physics, shadow, fusion, priority, geolocation incl. v2 + audit, association, persistence, ONNX integration, API endpoints, label purity, end-to-end) |
| `ruff check` + `ruff format --check` | clean |
| `mypy app` | clean except one pre-existing `physics.py:91` note |
| Frontend `tsc -b --noEmit` | clean |
| ESLint | 0 errors / 0 warnings |
| `npm run build` | successful |
| Real end-to-end | upload → analysis → persistence → restart → spatial API → map markers → Lab → reports verified on a live server |

---

## Repository layout

```
backend/
  app/                FastAPI app
    api/              route modules (surveys, frames, detections, spatial, jobs, reports, media, samples, dashboard, analyze)
    pipeline/         quality, preprocess, detect, segment, classify_natural, shadow, physics, anomaly, geolocate, dimensions, consistency, fusion, priority, runner
    reports/          CSV / JSON / GeoJSON / PDF renderers
    models/           ONNX weights + registry loader
    fallback/         deterministic heuristic fallbacks
    association.py    multi-frame association engine (Phase 2)
    targets.py        persistent-target orchestrator
    db.py             SQLite persistence layer
    schemas.py        Pydantic API contract (frontend types mirror these)
  scripts/            data prep, synthetic generation, evaluation, export tooling
  tests/              pytest suite (math + persistence + association + geolocation + e2e + API)
frontend/
  src/pages/          Dashboard, Upload, Lab, Explorer, Map, Hazards, Reports
  src/components/     TargetMap (shared MapLibre canvas), AmbientBackground, ui kit
  src/api.ts          typed API client (mirrors backend schemas)
models/
  registry.json       pluggable model registry (which backend serves which task)
  weights/            ONNX weight files
data/
  samples/            MD-FLS watertank subset + synthetic SSS demos (manifests included)
docs/                 architecture, association, geolocation, map, metrics, datasets, training, visual system
training/             Colab/GPU training notebook + packaging
```

---

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | system design deep-dive |
| [`docs/association.md`](docs/association.md) | Phase 2 association rules, thresholds, evidence records |
| [`docs/geolocation.md`](docs/geolocation.md) | Phase 3 conventions, provenance schema, uncertainty |
| [`docs/map.md`](docs/map.md) | Phase 4 map semantics, spatial API contract, honesty rules |
| [`docs/metrics.md`](docs/metrics.md) | fusion weights, thresholds, calibration discussion |
| [`docs/datasets.md`](docs/datasets.md) / [`docs/dataset-cards.md`](docs/dataset-cards.md) | training/demo data provenance & cards |
| [`docs/training.md`](docs/training.md) | training pipeline (Colab notebook) |
| [`docs/frontend-visual-system.md`](docs/frontend-visual-system.md) | design system, animation principles, accessibility |
| [`docs/demo-script.md`](docs/demo-script.md) | guided demo walkthrough |

---

## Known limitations (stated honestly)

- **Geolocation is approximate by design.** Derived positions come from flat-seabed sonar geometry with documented assumptions; uncertainty is expressed as a 2σ ellipse, not centimeter precision. No full 3D reconstruction is claimed.
- **Frame position ≠ object position.** The frame's GPS is a reference; the object offset is estimated from sonar geometry and can be wrong if the metadata is wrong.
- **Deep-link refresh** on frontend routes can hit a backend 404 in the single-port demo (known SPA-fallback limitation; client-side navigation is unaffected).
- **Prototype thresholds.** Association/fusion/decision thresholds are documented, tunable heuristics — not scientifically validated universal values.
- **Single-writer SQLite** suits the prototype's scale, not concurrent multi-user production load.
- **Taxonomy is deliberately narrow** (four classes) — the honest scope of the current training data.
- **SPA routing fallback** and other minor items are tracked in the docs above.

---

## Licensing & attribution

- `data/samples/md_fls_watertank*` is a small subset of the **Marine Debris Forward-Looking Sonar dataset** (Valdenegro-Toro et al., OCEANS 2025 Brest — doi:10.1109/OCEANS58557.2025.11104623). The source repository declares no explicit license; the subset is bundled for internal demo/evaluation only. Full cards: [`docs/dataset-cards.md`](docs/dataset-cards.md).
- Synthetic SSS imagery is generated by `backend/scripts/synth.py` and contains no real-world data.
- Model training tooling and dataset links: [`docs/training.md`](docs/training.md) and [`docs/datasets.md`](docs/datasets.md).
