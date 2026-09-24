"""Train YOLOv8 detection on the packaged sonar datasets (Colab GPU required).

- GPU is REQUIRED: exits with a clear error if torch.cuda.is_available() is False.
- Trains one model per data.yaml under bundle/detect/*, writes runs/<name>/.
- Prints and stores final val metrics verbatim from ultralytics — no editing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

B = Path("/content/aqua_train/bundle")
RUNS = Path("/content/aqua_train/runs")


def require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "\n" + "=" * 70 + "\nERROR: No GPU detected.\n"
            "In Colab: Runtime > Change runtime type > Hardware accelerator = GPU (T4),\n"
            "then re-run this cell. Detection training on CPU would take days and is\n"
            "deliberately not supported by this script.\n" + "=" * 70,
            file=sys.stderr,
        )
        raise SystemExit(3)
    name = torch.cuda.get_device_name(0)
    print(f"GPU OK: {name}")


def main() -> None:
    require_gpu()
    from ultralytics import YOLO

    cfg = yaml.safe_load((B / "config.yaml").read_text(encoding="utf-8"))
    dcfg = cfg["detection"]
    RUNS.mkdir(parents=True, exist_ok=True)

    yamls = sorted(B.glob("detect/*/data.yaml"))
    if not yamls:
        raise SystemExit("no data.yaml found — run prepare_detection.py first")
    results_all = {}
    for dy in yamls:
        name = dy.parent.name
        model = YOLO(dcfg["model"])
        model.train(
            data=str(dy),
            epochs=dcfg["epochs"],
            imgsz=dcfg["imgsz"],
            batch=dcfg["batch"],
            patience=dcfg["patience"],
            project=str(RUNS),
            name=f"yolo_{name}",
            seed=cfg["seed"],
            deterministic=True,
            # augmentation (sonar-conserving, see config.yaml)
            hsv_h=dcfg["aug"]["hsv_h"],
            hsv_s=dcfg["aug"]["hsv_s"],
            hsv_v=dcfg["aug"]["hsv_v"],
            degrees=dcfg["aug"]["degrees"],
            translate=dcfg["aug"]["translate"],
            scale=dcfg["aug"]["scale"],
            shear=dcfg["aug"]["shear"],
            flipud=dcfg["aug"]["flipud"],
            fliplr=dcfg["aug"]["fliplr"],
            mosaic=dcfg["aug"]["mosaic"],
            mixup=dcfg["aug"]["mixup"],
            copy_paste=dcfg["aug"]["copy_paste"],
            erasing=dcfg["aug"]["erasing"],
            exist_ok=True,
        )
        metrics = model.val()  # final honest numbers on the site-level val split
        m = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        results_all[name] = m
        print(f"\n== {name} val: {json.dumps(m)} ==\n")
        (RUNS / f"yolo_{name}" / "final_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    (RUNS / "detection_results.json").write_text(json.dumps(results_all, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
