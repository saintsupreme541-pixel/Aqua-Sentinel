"""Evaluate the repaired U-Net checkpoint, export ONNX, verify parity, and
exercise the app adapter contract (steps 8-10 of the repair plan).

Stages (run in order):
  eval      - threshold sweep (0.20..0.50) on the site-split val set with the
              tiled sliding-window protocol; saves overlays for the best thr
  export    - torch.onnx.export to data/work/unet_investigation/repair/unet-sss.onnx
  parity    - PT vs ONNX on identical tensors (max/mean abs diff)
  adapter   - ONNXSegmenter contract tests: 512x512, 522x862, 197x233,
              real AI4Shipwrecks frame, real MD-FLS frame

Usage: python repair_eval.py <stage>
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch  # module-level: tiled_probs uses it from any caller scope

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEG = ROOT / "data" / "work" / "unet_investigation" / "bundle_seg"
REPAIR = ROOT / "data" / "work" / "unet_investigation" / "repair"
OVERLAYS = REPAIR / "overlays"
CKPT = REPAIR / "unet_repair_best.pt"
ONNX_OUT = REPAIR / "unet-sss.onnx"
SEED = 20260905
TILE = 512
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

sys.path.insert(0, str(HERE.parent))
from train_segmentation import UNet, _read_gray, _read_mask  # noqa: E402


def val_split() -> list[str]:
    """Exact prepare_segmentation.py::site_split reproduction (stdlib RNG)."""
    smap = json.loads((SEG / "site_map.json").read_text(encoding="utf-8"))
    sites = sorted(set(smap.values()))
    rng = random.Random(SEED)
    rng.shuffle(sites)
    n_val = max(1, round(len(sites) * 0.15))
    val_sites = set(sites[:n_val])
    return [s for s, site in sorted(smap.items()) if site in val_sites]


def tiled_probs(model_or_sess, img: np.ndarray, is_torch: bool, device: str = "cpu") -> np.ndarray:
    h, w = img.shape
    probs = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    step = TILE * 3 // 4
    for y in range(0, max(h - TILE, 0) + 1, step):
        for x in range(0, max(w - TILE, 0) + 1, step):
            yt, xt = min(y, h - TILE), min(x, w - TILE)
            t = img[yt : yt + TILE, xt : xt + TILE].astype(np.float32)[None, None] / 255.0
            if is_torch:
                xi = torch.from_numpy(t).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model_or_sess(xi))[0, 0].float().cpu().numpy()
            else:
                lg = model_or_sess.run(None, {IN_NAME: t})[0][0, 0]
                p = 1.0 / (1.0 + np.exp(-lg))
            probs[yt : yt + TILE, xt : xt + TILE] += p
            count[yt : yt + TILE, xt : xt + TILE] += 1
    return probs / np.maximum(count, 1)


def pr_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    pp, tp = int(pred.sum()), int(gt.sum())
    return {
        "dice": 2 * inter / max(pp + tp, 1),
        "iou": inter / max(union, 1),
        "precision": inter / max(pp, 1),
        "recall": inter / max(tp, 1),
    }


def stage_eval() -> None:
    import torch

    ck = torch.load(CKPT, map_location="cpu")
    model = UNet(base=ck.get("base", 32))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    device = "cpu"
    va = val_split()
    print(f"val images: {len(va)}", flush=True)

    gts, prob_maps = [], []
    for s in va:
        img = _read_gray(SEG / "images" / f"{s}.png")
        msk = _read_mask(SEG / "masks" / f"{s}.png") > 0
        gts.append(msk)
        prob_maps.append(tiled_probs(model, img, True, device))
        print(f"  inferred {s}", flush=True)

    sweep = {}
    for thr in THRESHOLDS:
        agg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
        for pm, gt in zip(prob_maps, gts):
            m = pr_metrics(pm > thr, gt)
            for k in agg:
                agg[k] += m[k]
        sweep[str(thr)] = {k: round(v / len(gts), 4) for k, v in agg.items()}
        print(f"thr={thr:.2f}: {sweep[str(thr)]}", flush=True)

    best_thr = max(THRESHOLDS, key=lambda t: sweep[str(t)]["dice"])
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    for s, pm, gt in list(zip(va, prob_maps, gts))[:4]:
        img = _read_gray(SEG / "images" / f"{s}.png")
        vis = cv2.cvtColor(cv2.resize(img, (864, int(864 * img.shape[0] / img.shape[1]))), cv2.COLOR_GRAY2BGR)
        scale = vis.shape[1] / img.shape[1]
        pred = cv2.resize((pm > best_thr).astype(np.uint8), (vis.shape[1], vis.shape[0])) > 0
        gtv = cv2.resize(gt.astype(np.uint8), (vis.shape[1], vis.shape[0])) > 0
        vis[gtv & ~pred] = (0, 200, 0)     # GT only: green
        vis[pred & ~gtv] = (0, 0, 230)     # prediction only: red
        vis[pred & gtv] = (230, 220, 0)    # overlap: cyan-ish
        cv2.imwrite(str(OVERLAYS / f"{s}_pred_vs_gt.png"), vis)

    (REPAIR / "pt_threshold_sweep.json").write_text(
        json.dumps({"checkpoint": str(CKPT), "val_images": len(va), "sweep": sweep, "best_thr_by_dice": best_thr}, indent=2),
        encoding="utf-8",
    )
    print(f"BEST thr by val dice: {best_thr} -> {sweep[str(best_thr)]}")


def stage_export() -> None:
    import torch

    ck = torch.load(CKPT, map_location="cpu")
    model = UNet(base=ck.get("base", 32))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    # Static [1,1,512,512] — exactly the deployed contract the app adapter
    # was built against (registry input_size 512, adapter resizes in/out).
    torch.onnx.export(
        model,
        torch.zeros(1, 1, TILE, TILE),
        str(ONNX_OUT),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )
    print("exported:", ONNX_OUT, f"({ONNX_OUT.stat().st_size/1e6:.1f} MB)")


def stage_parity() -> None:
    import torch
    import onnxruntime as ort

    ck = torch.load(CKPT, map_location="cpu")
    model = UNet(base=ck.get("base", 32))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    global IN_NAME
    IN_NAME = sess.get_inputs()[0].name
    rng = np.random.default_rng(SEED)
    diffs_max, diffs_mean = [], []
    for i in range(5):
        if i < 3:
            t = rng.random((1, 1, TILE, TILE)).astype(np.float32)
        else:  # real image tiles
            s = val_split()[0]
            img = _read_gray(SEG / "images" / f"{s}.png")
            h, w = img.shape
            y, x = rng.integers(0, h - TILE), rng.integers(0, w - TILE)
            t = img[y : y + TILE, x : x + TILE].astype(np.float32)[None, None] / 255.0
        with torch.no_grad():
            pt = model(torch.from_numpy(t)).numpy()
        ox = sess.run(None, {IN_NAME: t})[0]
        d = np.abs(pt - ox)
        diffs_max.append(float(d.max()))
        diffs_mean.append(float(d.mean()))
    print(f"parity: max abs diff {max(diffs_max):.3e} | mean abs diff {max(diffs_mean):.3e}")
    (REPAIR / "onnx_parity.json").write_text(
        json.dumps({"max_abs_diff_max": max(diffs_max), "mean_abs_diff_max": max(diffs_mean)}, indent=2), encoding="utf-8"
    )


def stage_adapter() -> None:
    sys.path.insert(0, str(ROOT / "backend"))
    import onnxruntime as ort
    from app.pipeline.segment import ONNXSegmenter

    cases: list[tuple[str, np.ndarray]] = [("uniform_512x512", (np.random.default_rng(1).random((512, 512)) * 255).astype(np.uint8))]
    cases.append(("synthetic_522x862", (np.random.default_rng(2).random((522, 862)) * 255).astype(np.uint8)))
    cases.append(("synthetic_197x233", (np.random.default_rng(3).random((197, 233)) * 255).astype(np.uint8)))
    ai4 = sorted((SEG / "images").glob("*.png"))[0]
    cases.append((f"ai4_{ai4.stem}", cv2.imread(str(ai4), cv2.IMREAD_GRAYSCALE)))
    mdfls = sorted((ROOT / "data" / "samples" / "md_fls_watertank").glob("*.png"))
    if mdfls:
        cases.append((f"mdfls_{mdfls[0].stem}", cv2.imread(str(mdfls[0]), cv2.IMREAD_GRAYSCALE)))

    seg = ONNXSegmenter(str(ONNX_OUT))
    results = {}
    for name, img in cases:
        out = seg.segment(img)
        mask = np.asarray(out["mask"]) if isinstance(out, dict) else np.asarray(out)
        results[name] = {"input_shape": list(img.shape), "mask_shape": list(mask.shape), "binary": bool(np.isin(mask, [0, 1]).all()), "positive_frac": float((mask > 0).mean())}
        print(name, results[name], flush=True)
    (REPAIR / "adapter_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "eval":
        stage_eval()
    elif stage == "export":
        stage_export()
    elif stage == "parity":
        stage_parity()
    elif stage == "adapter":
        stage_adapter()
    else:
        print("usage: repair_eval.py eval|export|parity|adapter")
        sys.exit(1)
