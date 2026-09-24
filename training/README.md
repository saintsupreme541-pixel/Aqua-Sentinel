# AQUA-SENTINEL — GPU training pipeline (Colab)

Training is **separate from the app**. The app stays CPU-first with heuristic
fallbacks; trained ONNX weights are optional drop-ins via `models/registry.json`.

## What trains on what (real data only — no invented labels)

| Task | Data (all local, verified) | Model | Export |
|---|---|---|---|
| Detection | MD-FLS watertank 1,868 img / 3,562 boxes (320 Chain excluded as hard negatives) · AI4Shipwrecks 261 img / 954 boxes · SCTD 357 img / 328 boxes (35 `human` excluded — a diver is not debris) | YOLOv8n @640 | `yolo-sss.onnx` |
| Segmentation | AI4Shipwrecks 161 binarized wreck masks (site-split) | U-Net 512-tiles | `unet-sss.onnx` |
| Nat-vs-art | 5,621 crops (4,540 artificial / 1,081 natural — 35 SCTD `human` crops excluded, review v3) | 64×64 CNN | `natart-sss.onnx` |

`ghost_net`, `pipe` and `cylinder` have **no verified real training data and
are deliberately not trained** — the detector never claims them; at inference,
objects that fit no supervised class surface as *Unknown Anomaly — Human
Review Required*. MD-FLS `Tire` keeps its **own supervised class** `tire`
(toroidal object — never merged into `cylinder`). See the semantic review v3
in `docs/dataset-cards.md` and the readiness report in `docs/training.md`.

## Local (CPU): build the bundle

```bash
make train-bundle     # → aqua_train_bundle.zip (~2.0 GB)
```

## Colab (GPU): train

1. Open `training/aqua_training.ipynb` in Colab (upload it at
   <https://colab.research.google.com>).
2. Runtime → Change runtime type → **T4 GPU**.
3. Run cells top-to-bottom. Cell 1 hard-fails without GPU. Cell 2 asks for
   `aqua_train_bundle.zip`.
4. Cell 10 downloads `aqua_onnx_models.zip` + `onnx_evaluation.json`.

## Local: wire the trained weights

```bash
cd models && unzip ../aqua_onnx_models.zip -d weights
```

`models/registry.json`:

```json
{
  "detection":   {"backend": "onnx", "path": "models/weights/yolo-sss.onnx",
                  "classes": ["wreck","debris","structure","tire"],
                  "input_size": 640, "conf_threshold": 0.15, "nms_iou": 0.45, "format": "v8"},
  "segmentation":{"backend": "onnx", "path": "models/weights/unet-sss.onnx", "num_classes": 1},
  "classifier":  {"backend": "onnx", "path": "models/weights/natart-sss.onnx"}
}
```

Restart uvicorn. Dashboard → each detection now reports `backend_used:
{detection: onnx, …}`; delete a weight file and the heuristic fallback takes
over automatically (verified by the app's smoke test).

## Files

- `package_for_colab.py` — stage 1 bundle builder (local)
- `prepare_detection.py` / `prepare_segmentation.py` / `prepare_classifier.py` — stage 2, leak-safe splits
- `train_detection.py` / `train_segmentation.py` / `train_classifier.py` — stage 3, GPU training
- `export_onnx.py` — stage 4, app-contract exports + numeric verification
- `evaluate_onnx.py` — stage 5, independent metrics on held-out data
- `config.yaml` — all hyperparameters + sonar-conserving augmentation
- `requirements-colab.txt`
- `aqua_training.ipynb` — the one-click notebook
