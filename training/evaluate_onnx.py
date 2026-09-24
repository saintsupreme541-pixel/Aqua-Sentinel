"""Independent evaluation of exported ONNX models on held-out test data.

Runs the exported ONNX files exactly like the app does (same letterbox, NMS,
thresholds, mask binarization) and reports final metrics verbatim:

- detection:  P/R/F1 @ IoU 0.5 vs ground-truth boxes (same matcher as the
              app-side backend/scripts/evaluate.py)
- segmenter:  Dice/IoU on val-site whole images
- classifier: accuracy / balanced accuracy / ROC-AUC on the val crop split

This script is the "trust but verify" gate before weights enter the app.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

B = Path("/content/aqua_train/bundle")
OUT = Path("/content/aqua_train/runs/onnx")


# ---------------------------------------------------------------- detection
def letterbox(img: np.ndarray, size: int = 640):
    import cv2

    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size), 114, np.uint8)
    canvas[pady : pady + nh, padx : padx + nw] = r
    return canvas, (s, padx, pady)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ar = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (ar[i] + ar[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def detect(session, gray: np.ndarray, nc: int, conf=0.15, iou_thr=0.45):
    inp, (s, px, py) = letterbox(gray)
    blob = inp[None, None].astype(np.float32) / 255.0
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    t = np.asarray(out)
    if t.ndim == 3 and t.shape[1] == 4 + nc:
        t = t[0].T
    elif t.ndim == 3 and t.shape[2] == 4 + nc:
        t = t[0]
    else:
        raise ValueError(f"unexpected det output {t.shape}")
    sc = t[:, 4 : 4 + nc].max(1)
    ci = t[:, 4 : 4 + nc].argmax(1)
    cand = np.where(sc >= conf)[0]
    if not len(cand):
        return []
    xy = np.array(
        [[t[i, 0] - t[i, 2] / 2, t[i, 1] - t[i, 3] / 2, t[i, 0] + t[i, 2] / 2, t[i, 1] + t[i, 3] / 2] for i in cand]
    )
    keep = nms(xy, sc[cand], iou_thr)
    # inverse letterbox exactly like the app: subtract padding FIRST, then
    # divide by scale — ((x - pad) / s), not (x / s - pad).
    return [((xy[k] - [px, py, px, py]) / s, int(ci[cand][k]), float(sc[cand][k])) for k in keep]


def eval_detection(iou_match: float = 0.5) -> dict:
    import cv2

    sess = ort.InferenceSession(str(OUT / "yolo-sss.onnx"), providers=["CPUExecutionProvider"])
    classes = (B / "detect" / "ai4shipwrecks" / "classes.txt").read_text().split()
    tp = fp = fn = 0
    test_imgs = sorted((B / "detect" / "ai4shipwrecks" / "test" / "images").glob("*.png"))
    if not test_imgs:
        return {"note": "no ai4 test split (bundled as train-only); val metrics from training are the reference"}
    for img_p in test_imgs:
        lbl = B / "detect" / "ai4shipwrecks" / "test" / "labels" / (img_p.stem + ".txt")
        gts = []
        if lbl.exists():
            for ln in lbl.read_text().strip().splitlines():
                c, cx, cy, w, h = ln.split()
                gts.append([float(v) for v in (cx, cy, w, h)])
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        dets = detect(sess, img, len(classes))
        matched = [False] * len(gts)
        for xy, _ci, _s in dets:
            x1, y1, x2, y2 = xy
            best, bi = 0.0, -1
            for gi, (gcx, gcy, gw, gh) in enumerate(gts):
                gx1, gy1, gx2, gy2 = (gcx - gw / 2) * w, (gcy - gh / 2) * h, (gcx + gw / 2) * w, (gcy + gh / 2) * h
                ix1, iy1 = max(x1, gx1), max(y1, gy1)
                ix2, iy2 = min(x2, gx2), min(y2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                u = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter
                v = inter / max(u, 1e-9)
                if v > best:
                    best, bi = v, gi
            if best >= iou_match and bi >= 0 and not matched[bi]:
                matched[bi] = True
                tp += 1
            else:
                fp += 1
        fn += matched.count(False)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "precision@0.5": round(p, 4),
        "recall@0.5": round(r, 4),
        "f1@0.5": round(2 * p * r / max(p + r, 1e-9), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "images": len(test_imgs),
    }


# --------------------------------------------------------------- segmenter
def eval_seg() -> dict:
    import cv2

    sess = ort.InferenceSession(str(OUT / "unet-sss.onnx"), providers=["CPUExecutionProvider"])
    seg = B / "seg" / "ai4shipwrecks"
    split = json.loads((seg / "split.json").read_text())
    inter = union = pp = tp_ = 0
    tile = 512
    for s in split["val"]:
        img = cv2.imread(str(seg / "images" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(seg / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        msk = msk > 0
        h, w = img.shape
        probs = np.zeros((h, w), np.float32)
        cnt = np.zeros((h, w), np.float32)
        step = int(tile * 0.75)
        ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
        for y in ys:
            for x in xs:
                yt, xt = min(y, h - tile), min(x, w - tile)
                t = img[yt : yt + tile, xt : xt + tile][None, None].astype(np.float32) / 255.0
                p = sess.run(None, {sess.get_inputs()[0].name: t})[0]
                p = 1 / (1 + np.exp(-np.asarray(p)))[0, 0]
                probs[yt : yt + tile, xt : xt + tile] += p
                cnt[yt : yt + tile, xt : xt + tile] += 1
        pred = (probs / np.maximum(cnt, 1)) > 0.5
        inter += int((pred & msk).sum())
        union += int((pred | msk).sum())
        pp += int(pred.sum())
        tp_ += int(msk.sum())
    dice = 2 * inter / max(pp + tp_, 1)
    return {
        "dice": round(float(dice), 4),
        "iou": round(float(inter / max(union, 1)), 4),
        "val_images": len(split["val"]),
    }


# --------------------------------------------------------------- classifier
def eval_cls() -> dict:
    import cv2
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    sess = ort.InferenceSession(str(OUT / "natart-sss.onnx"), providers=["CPUExecutionProvider"])
    root = B / "cls" / "split" / "val"
    ys, ps = [], []
    for label, idx in (("natural", 0), ("artificial", 1)):
        for p in sorted((root / label).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            x = cv2.resize(img, (64, 64)).astype(np.float32)[None, None] / 255.0
            out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
            e = np.exp(np.asarray(out)[0] - np.asarray(out)[0].max())
            ps.append(float(e[1] / e.sum()))
            ys.append(idx)
    preds = [int(p > 0.5) for p in ps]
    return {
        "accuracy": round(float(np.mean(np.array(preds) == np.array(ys))), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(ys, preds)), 4),
        "roc_auc": round(float(roc_auc_score(ys, ps)), 4),
        "n": len(ys),
    }


def main() -> None:
    res = {}
    if (OUT / "yolo-sss.onnx").exists():
        print("evaluating detector on AI4 test split…")
        res["detection_ai4_test"] = eval_detection()
    if (OUT / "unet-sss.onnx").exists():
        print("evaluating segmenter on val sites…")
        res["segmentation_val"] = eval_seg()
    if (OUT / "natart-sss.onnx").exists():
        print("evaluating classifier on val crops…")
        res["classifier_val"] = eval_cls()
    out = Path("/content/aqua_train/runs/onnx_evaluation.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
