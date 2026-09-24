"""Export trained YOLO weights to ONNX for CPU inference (optional step).

The registry can consume either ``.pt`` (requires ``pip install ultralytics``)
or ONNX (CPU via onnxruntime, no torch needed at serve time).  This helper
exports a trained model and prints the exact registry snippet to paste into
``models/registry.json``.

Usage::

    python scripts/export_onnx.py path/to/best.pt --imgsz 640 --out models/detector.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path, help="trained ultralytics .pt file")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("models/detector.onnx"))
    ap.add_argument("--task", choices=["detection", "segmentation"], default="detection")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed — install with `pip install ultralytics` first") from None

    model = YOLO(str(args.weights))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True)
    exported = args.out.with_suffix(".onnx") if args.out.suffix != ".onnx" else args.out
    # ultralytics writes next to the source by default; move if needed
    alt = args.weights.with_suffix(".onnx")
    if alt.exists() and not exported.exists():
        alt.replace(exported)
    if not exported.exists():
        raise SystemExit(f"expected export at {exported} but it was not produced")

    classes = list(model.names.values()) if hasattr(model, "names") else ["debris"]
    snippet = {
        args.task: {
            "backend": "onnx",
            "path": str(exported),
            "classes": classes,
            "input_size": args.imgsz,
            "conf_threshold": 0.25,
            "nms_iou": 0.45,
            "format": "v8",
        }
    }
    print(f"exported → {exported}")
    print("paste into models/registry.json:")
    print(json_dumps(snippet))


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    main()
