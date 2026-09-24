# DATASET READINESS REPORT — SIH26057 training

*Generated from on-disk verification, not from papers. Every count below was
computed by script over the actual files in this repository (see
`backend/scripts/convert_datasets.py`, `training/package_for_colab.py` and the
audit runs of 2026-09-05).*

## 1. Datasets found (local, verified)

| Dataset | Modality | Images | Boxes | Masks | Integrity |
|---|---|---|---|---|---|
| MD-FLS watertank-segmentation | FLS (tank, ARIS 3000) | 1,868 | 3,882 (VOC XML) | 1,868 | 0 corrupt, perfect 3-way name alignment |
| AI4Shipwrecks | **SSS (field, NOAA Thunder Bay)** | 261 labelled + 25 terrain negatives | 954 (derived from masks) | 161 wreck instances | zip CRC OK, sizes match paper (286 total) |
| SCTD 1.0 | mixed SSS/FLS/SAS | 357 | 363 (VOC XML) | — | 357/357 XML↔image alignment |
| SeabedObjects-KLSG | **SSS (field)** | 447 (385 ship, 62 plane) | — (classification only) | — | count matches README claim |
| UATD | MFLS | **download in progress** (figshare 202-preparing) | ~9,000 expected | — | pending |
| Synthetic SSS + MD-FLS 24/8 subsets | demo | bundled under `data/samples/` | yes | yes | all manifests resolve |

Modality note: the problem statement is side-scan sonar. Today's **real-SSS
pixel/box-labelled** data is AI4Shipwrecks (wrecks only) + KLSG (classification
only). MD-FLS/SCTD/UATD are FLS or mixed — valuable but a documented domain
gap that validation on AI4 test sites partially measures.

## 2. Taxonomy decision log (semantic label review v3)

Every source mapping was reviewed against the actual annotations — objects
were NOT merged into a class merely because both are man-made. Rejected
mappings and the reason:

| Source label | Source | Count | Final class | Decision rationale |
|---|---|---|---|---|
| AI4 binary masks | AI4Shipwrecks | 954 boxes / 161 masks | `wreck` | sunken vessels, real SSS, expert pixel labels |
| ship / aircraft | SCTD | 271 / 57 | `wreck` | sunken ships / aircraft — same wreck semantic, real sonar |
| **Tire** | MD-FLS | 615 | **`tire`** | **v3: rejected `cylinder`** — a tire is a toroidal/ring-shaped manufactured object, not a generic cylinder; sonar resemblance does not establish semantic equivalence for supervised detection. Kept as its own supervised class; never used to support `cylinder`. |
| Propeller | MD-FLS | 197 | `structure` | **rejected `wreck`**: intact man-made hard structure is not a sunken-vessel semantic; kept as its own class |
| Wall | MD-FLS | 1,056 | `structure` | tank wall = background hard-negative, distinct from debris |
| Bottle/Can/Drink-carton/Hook/Shampoo-bottle/Standing-bottle/Valve | MD-FLS | 1,694 | `debris` | generic man-made debris; no closer evidence-backed class (v3 count corrected: 468+347+292+240+171+111+65, verified per-XML) |
| **human** | SCTD | 35 | **excluded from supervision** | **v3: rejected `debris`** — a human/diver is not marine debris. Boxes removed from detector training; `raw_class` preserved; usable only as contextual/hard-negative examples or a future separate class. |
| **Chain** | MD-FLS | 320 | **hard-negative only** | **rejected `net_like`/`ghost_net`**: rigid metal links ≠ flexible filament mesh — different geometry, texture and echo signature. Used only for nat-vs-art crops + detector hard negatives |
| **ghost_net** | — | 0 | **not trained** | no genuine labelled examples anywhere; never claimed by the detector |
| **pipe** | — | 0 | **not trained** | no genuine labelled examples anywhere; never claimed by the detector |
| **cylinder** | — | **0** | **not trained** | **v3: 0 verified examples** — the 615 Tire annotations were the only candidate support and are explicitly rejected (toroidal ≠ solid cylinder); class stays in `unsupported_classes` until genuine cylinder annotations exist |

**Final supervised detector classes: `wreck, debris, structure, tire`**
(fixed order = `data/training/*/yolo/classes.txt` = `models/registry.json`).
`ghost_net`, `pipe` and `cylinder` are absent by design: objects that fit no
supervised class surface through the **Unknown-Anomaly pathway** ("Unknown
Anomaly — Human Review Required") instead of being force-fitted into a
lookalike class. The synthetic-SSS generator can render net/pipe-like objects
for UI demos, but synthetic data is never used for supervised weights and any
metric on it must be labelled synthetic. Hard negatives retained: 1,056
`Wall` boxes + 25 AI4 terrain images (SSS) + 100 empty-label frames. The 35
SCTD `human` boxes are excluded from ALL supervision (detector and
classifier crops); they remain available for a future separate diver class
or hard-negative mining if ever justified — never as `debris`.

## 3. Annotation formats

- VOC XML (MD-FLS, SCTD) → converted to YOLO `.txt` + backend manifests
- Binary/instance mask PNG (AI4Shipwrecks, MD-FLS) → connected components → boxes; binarized for U-Net
- Folder-as-label (KLSG) → classification manifest only

## 4. Usable samples (packaged bundle, verified by build output)

- Detection: 2,486 images / 4,844 boxes (MD-FLS 1,868 img / 3,562 boxes — 320 Chain excluded as hard negatives — + AI4 261 / 954 + SCTD 357 / 328 — 35 `human` boxes excluded: a diver is not debris)
- Segmentation: 161 image/mask pairs (site-split 137/24)
- Classification: 5,621 crops (4,540 artificial / 1,081 natural — 35 SCTD `human` crops excluded, review v3)

## 5. Class imbalance (measured)

- MD-FLS: Wall 1,056 · Tire 615 · Bottle 468 · … · Standing-bottle 65 (~16:1)
- AI4: single class (wreck)
- Classifier: artificial:natural = 4.2:1 (handled by weighted loss + balanced-accuracy metric)

## 6. Split design (leak-safe, implemented in `training/prepare_*.py`)

- **Site-level grouping**: AI4 keeps every frame of a wreck site in one split
  (25 sites total; detection val = 2 sites carved from the train-image set,
  segmentation val = 4 sites).
- **Chronological-tail fallback** for single-session datasets (MD-FLS tank,
  SCTD) — used and **disclosed** in `split_report.json`, because adjacent
  pings overlap; val numbers for those sets are optimistic and any claims must
  say so.
- Classifier: group-wise by source image/site — verified zero site-straddling
  by assertion in the smoke test.

Verified deterministic split counts (seed 20260905, replicated locally from
the shipped manifests — images / boxes):

| Task · dataset | Train | Val | Test |
|---|---|---|---|
| Detect · AI4Shipwrecks (SSS, site-level) | 111 img / 583 wreck | 30 img / 42 wreck | 120 img / 329 wreck (dataset's own site split) |
| Detect · SCTD (tail fallback, disclosed) | 303 img / 272 wreck | 54 img / 56 wreck | — |
| Detect · MD-FLS (tail fallback, disclosed) | 1,587 img — structure 1,078 · debris 1,440 · tire 520 | 281 img — 175 / 254 / 95 | — |
| Segment · AI4Shipwrecks (site-level) | 145 masks | 16 masks | — (val is held-out sites) |
| Classifier (2,219 site groups → 333 val) | 4,816 crops (3,900 art / 916 nat) | 805 crops (640 / 165) | — |

## 7. Risks (ranked)

1. **Missing classes** (`pipe`, `ghost_net`, `cylinder`) — detector will not
   emit them; fusion/anomaly stages must cover them, or synthetic-only
   training must be labeled as such.
2. **Domain gap** FLS-tank → SSS-field; mitigated by training a dedicated
   AI4 (SSS) model + mixed model, and by reporting per-dataset val numbers.
3. **Waterfall aspect ratios** (1728×up-to-18k) — handled by tiling, but
   tile-boundary truncation of wrecks slightly depresses recall.
4. **SCTD license** and MD-FLS license unverified for public release —
   fine for SIH demo/training, flagged in dataset cards.
5. UATD pending; adds scale but is FLS — treat as augmentation, not headline.

## 8. Exact next training command

Local packaging (done): `make train-bundle` → `aqua_train_bundle.zip`.

On Colab (T4): open `training/aqua_training.ipynb`, run cells 1→10. Cell 1
hard-fails without GPU. The single-stage CLI equivalent on a GPU machine:

```bash
cd /content/aqua_train
python training/prepare_detection.py && python training/prepare_segmentation.py --emit-filelists && python training/prepare_classifier.py
python training/train_detection.py && python training/train_segmentation.py && python training/train_classifier.py
python training/export_onnx.py && python training/evaluate_onnx.py
```

Output: `runs/aqua_onnx_models.zip` + `runs/onnx_evaluation.json` (verbatim
metrics). Wire-in: unzip to `models/weights/`, flip
`models/registry.json` backends to `onnx` (exact JSON in `training/README.md`),
restart uvicorn. No-metrics policy: nothing in this repo prints a number that
was not computed by `evaluate_onnx.py` / `model.val()`.
