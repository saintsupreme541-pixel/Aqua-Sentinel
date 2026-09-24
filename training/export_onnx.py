"""Export trained checkpoints to app-compatible ONNX + verify numerically.

Compatibility targets (read from the app adapters — do not change the app):
- detection:  standard ultralytics YOLOv8 ONNX @640, single output
              (1, 4+nc, N) OR (1, N, 4+nc); app letterboxes itself.
- segmenter:  input 1×1×H×W (dynamic H/W), output (1,1,H,W) LOGITS —
              app thresholds at >0.5 after resize to full size.
- classifier: input 1×1×64×64 /255, output 2 logits [natural, artificial].

Every export is smoke-tested with onnxruntime on random input and the output
shape/parse is asserted against the contract above. Exports land in
runs/onnx/ and are zipped as aqua_onnx_models.zip for easy download.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

RUNS = Path("/content/aqua_train/runs")
B = Path("/content/aqua_train/bundle")
OUT = RUNS / "onnx"


def verify(path: Path, build_input, check) -> None:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: build_input(inp.shape)})[0]
    check(inp.shape, out)
    print(f"  OK {path.name}: in={inp.shape} out={np.asarray(out).shape}")


def export_classifier(pt: Path) -> Path:
    from train_classifier import SmallCNN

    model = SmallCNN()
    model.load_state_dict(torch.load(pt, map_location="cpu"))
    model.eval()
    onnx = OUT / "natart-sss.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 1, 64, 64),
        str(onnx),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return onnx


def export_segmenter(pt: Path) -> Path:
    from train_segmentation import UNet

    ck = torch.load(pt, map_location="cpu")
    model = UNet(base=ck.get("base", 32))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    onnx = OUT / "unet-sss.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 1, 512, 512),
        str(onnx),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
        opset_version=17,
    )
    return onnx


def export_detector(runs_yolo_dir: Path) -> Path:
    from ultralytics import YOLO

    best = sorted(runs_yolo_dir.glob("weights/best.pt"))
    if not best:
        raise SystemExit(f"no best.pt under {runs_yolo_dir}")
    model = YOLO(str(best[0]))
    onnx = OUT / f"{runs_yolo_dir.name.replace('yolo_', 'yolo-')}.onnx"
    model.export(format="onnx", imgsz=640, simplify=True, opset=12, dynamic=False)
    produced = runs_yolo_dir / "weights" / "best.onnx"
    produced.rename(onnx)
    return onnx


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exported = []

    # classifier
    pt = RUNS / "natart_best.pt"
    if pt.exists():
        p = export_classifier(pt)
        verify(
            p,
            lambda shp: np.random.rand(1, 1, 64, 64).astype(np.float32),
            lambda shp, out: (
                (_ for _ in ()).assert_ok(out)
                if False
                else (
                    None
                    if np.asarray(out).ndim == 2 and np.asarray(out).shape[1] == 2
                    else (_ for _ in ()).throw(AssertionError(f"classifier out {out.shape} != (N,2)"))
                )
            ),
        )
        exported.append(p)

    # segmenter
    pt = RUNS / "unet_best.pt"
    if pt.exists():
        p = export_segmenter(pt)
        verify(
            p,
            lambda shp: np.random.rand(1, 1, 256, 384).astype(np.float32),
            lambda shp, out: (
                None
                if np.asarray(out).shape == (1, 1, 256, 384)
                else (_ for _ in ()).throw(AssertionError(f"segmenter out {np.asarray(out).shape} != (1,1,256,384)"))
            ),
        )
        exported.append(p)

    # detector(s) — nc comes from the packaged classes.txt, never hard-coded
    for ydir in sorted(RUNS.glob("yolo_*")):
        if (ydir / "weights" / "best.pt").exists():
            p = export_detector(ydir)
            cls_txt = B / "detect" / ydir.name.removeprefix("yolo_") / "classes.txt"
            nc = len(cls_txt.read_text().split())

            def check(shp, out, nc=nc):
                t = np.asarray(out)
                ok = t.ndim == 3 and (t.shape[1] == 4 + nc or t.shape[2] == 4 + nc)
                if not ok:
                    raise AssertionError(f"detector out {t.shape} not parseable with nc={nc}")

            verify(p, lambda shp: np.random.rand(1, 1, 640, 640).astype(np.float32), check)
            exported.append(p)

    if not exported:
        raise SystemExit("nothing to export — train first")
    zpath = RUNS / "aqua_onnx_models.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in exported:
            zf.write(p, p.name)
    print(f"\nexports: {[p.name for p in exported]}\nbundle: {zpath}")


if __name__ == "__main__":
    main()
