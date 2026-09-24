"""Evaluation harness for the AQUA-SENTINEL fallback (heuristic) baseline.

Runs the *actual* detection/segmentation code used by the API against
ground-truth manifests (synthetic SSS + the real MD-FLS watertank subset)
and reports honest metrics:

- P / R / F1 @ IoU 0.5  (any-object level: the heuristic detector cannot
  claim per-class recognition beyond net-like tagging, so we do not)
- false alarms per image (false-alarm rate)
- mean Dice between the heuristic highlight mask and the pixel ground truth
  (real-data subset only; class-agnostic, binarised)
- per-set breakdown so "synthetic vs real" performance is never conflated

This is the *fallback baseline* used when no trained weights are present.
Drop ONNX weights into models/ and rerun to compare a real model against it.

Usage::

    python scripts/evaluate.py [--manifest data/samples/synthetic_sss.json]
                               [--manifest data/samples/md_fls_watertank.json]
                               [--samples-root data/samples]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.fallback.segment_threshold import segment_heuristic  # noqa: E402
from app.models.registry import load_registry  # noqa: E402
from app.pipeline import detect, preprocess  # noqa: E402


def iou(a: dict, b: dict) -> float:
    ax0, ay0 = a["x"], a["y"]
    bx0, by0 = b["x"], b["y"]
    ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
    bx1, by1 = bx0 + b["w"], by0 + b["h"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def match(dets: list[dict], gts: list[dict], thr: float = 0.5):
    """Greedy IoU matching → (tp, fp, fn). GT boxes are flat {x,y,w,h}."""
    gt_boxes = [{"x": g["x"], "y": g["y"], "w": g["w"], "h": g["h"]} for g in gts]
    tp = 0
    fp = 0
    used = set()
    for d in sorted(dets, key=lambda x: -x["score"]):
        best_i, best = None, 0.0
        for j, gb in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(d["box"], gb)
            if v > best:
                best, best_i = v, j
        if best_i is not None and best >= thr:
            tp += 1
            used.add(best_i)
        else:
            fp += 1
    fn = len(gts) - len(used)
    return tp, fp, fn


def load_binarized_mask(path: Path) -> np.ndarray:
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    return 2 * inter / max(float(a.sum() + b.sum()), 1.0)


def evaluate_manifest(manifest_path: Path, samples_root: Path) -> dict:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = samples_root / man.get("images_root", manifest_path.stem)
    registry = load_registry()
    _det, backend, warn = detect.get_detection_backend(registry)
    if backend != "heuristic":
        print(f"  ! registry resolved detection to '{backend}' — evaluating that backend, not the heuristic baseline")

    tp = fp = fn = 0
    fa_images = 0
    dices: list[float] = []
    per_image: list[dict] = []
    for gt in man.get("ground_truth", []):
        img_p = root / gt["image"]
        if not img_p.exists():
            print(f"  ! missing image {img_p} (skipped)")
            continue
        gray = preprocess.load_grayscale(img_p)
        meta = {**man.get("survey_meta", {}), **gt.get("meta", {})}
        processed, _ = preprocess.preprocess(gray, meta, preset=meta.get("preprocess_preset", "light"))
        dets = _det.detect(processed, meta)
        boxes = gt["boxes"]
        t, f, n = match(dets, boxes)
        tp += t
        fp += f
        fn += n
        if f > 0:
            fa_images += 1
        per_image.append(
            {
                "image": gt["image"],
                "gt_objects": len(boxes),
                "detections": len(dets),
                "tp": t,
                "fp": f,
                "fn": n,
            }
        )

        # mask Dice (class-agnostic) — segment each GT object, compare to pixel GT
        mask_p = root / "Masks" / Path(gt["image"]).name
        if mask_p.exists():
            gt_mask = load_binarized_mask(mask_p)
            if gt_mask is not None:
                for b in boxes:
                    seg, _ = segment_heuristic(processed, b)
                    dices.append(dice((seg > 0).astype(np.uint8), gt_mask))

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    n_images = len(per_image)
    return {
        "manifest": man["id"],
        "name": man["name"],
        "backend": backend,
        "images": n_images,
        "gt_objects": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_alarms_per_image": round(fp / max(n_images, 1), 3),
        "images_with_false_alarms": fa_images,
        "mean_dice": round(float(np.mean(dices)), 4) if dices else None,
        "per_image": per_image,
        "warning": warn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", default=[], help="manifest path (repeatable)")
    ap.add_argument(
        "--samples-root", type=Path, default=Path(__file__).resolve().parent.parent.parent / "data" / "samples"
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.manifest:
        args.manifest = [
            str(args.samples_root / "synthetic_sss.json"),
            str(args.samples_root / "md_fls_watertank.json"),
        ]

    print("=" * 74)
    print("AQUA-SENTINEL evaluation — fallback baseline (no trained weights)")
    print("=" * 74)
    results = []
    for m in args.manifest:
        mp = Path(m)
        print(f"\nEvaluating {mp.name}:")
        r = evaluate_manifest(mp, args.samples_root)
        results.append(r)
        print(f"  backend          : {r['backend']}")
        print(f"  images           : {r['images']}   gt objects: {r['gt_objects']}")
        pr = f"  P / R / F1 @IoU.5: {r['precision']:.3f} / {r['recall']:.3f} / {r['f1']:.3f}"
        print(f"{pr}   (tp {r['tp']}, fp {r['fp']}, fn {r['fn']})")
        fa = f"  false alarms/img : {r['false_alarms_per_image']:.3f}"
        print(f"{fa}  ({r['images_with_false_alarms']}/{r['images']} images)")
        if r["mean_dice"] is not None:
            print(f"  mask Dice (mean) : {r['mean_dice']:.3f}  (real pixel-GT subset)")
        if r["warning"]:
            print(f"  note             : {r['warning']}")

    total = {
        "tp": sum(r["tp"] for r in results),
        "fp": sum(r["fp"] for r in results),
        "fn": sum(r["fn"] for r in results),
    }
    p = total["tp"] / max(total["tp"] + total["fp"], 1)
    r_ = total["tp"] / max(total["tp"] + total["fn"], 1)
    print("\n" + "-" * 74)
    f1 = 2 * p * r_ / max(p + r_, 1e-9)
    head = f"COMBINED  P / R / F1 @IoU.5 = {p:.3f} / {r_:.3f} / {f1:.3f}"
    print(f"{head}  (tp {total['tp']}, fp {total['fp']}, fn {total['fn']})")
    print("Note: any-object level. The heuristic baseline does not claim per-class")
    print("recognition; net-like tagging is reported as a separate qualitative stage.")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"combined": total, "sets": results}, indent=2), encoding="utf-8")
        print(f"wrote JSON metrics -> {args.json_out}")


if __name__ == "__main__":
    main()
