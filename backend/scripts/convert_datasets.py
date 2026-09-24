"""Convert raw public sonar datasets into AQUA-SENTINEL training format.

For each dataset produces under ``data/training/<id>/``:

* ``manifest.json``   — backend sample-manifest schema (images + ground_truth
  boxes in absolute pixels, taxonomy classes, ``raw_class`` preserved).
* ``yolo/``           — YOLO detection export (images/ + labels/*.txt +
  classes.txt) for Ultralytics training; empty/negative images get an empty
  labels file so YOLO learns background.
* ``masks/``          — (AI4Shipwrecks only) binary segmentation masks
  0/1 uint8 PNGs aligned with images, for U-Net training.

Supervised taxonomy (backend registry order)::

    0 wreck  1 debris  2 structure  3 tire

ONLY classes with genuine labelled examples in the source data (semantic
review v3 in docs/dataset-cards.md). ``ghost_net`` / ``pipe`` / ``cylinder``
are NOT classes: no real labelled examples exist in any local dataset, so
the detector must not claim them — unsupported objects route through the
Unknown-Anomaly pathway at inference time instead of being force-fitted
into a lookalike.

Mapping decisions (documented in docs/dataset-cards.md):
* AI4Shipwrecks masks -> connected components -> ``wreck`` boxes.
* SCTD VOC: ship/aircraft -> ``wreck`` (sunken ships / aircraft);
  human -> EXCLUDED from detector supervision (a diver is not marine
  debris); raw_class stays in the manifest for traceability.
* MD-FLS: Tire -> ``tire`` (toroidal object, kept its own supervised class
  — never merged into ``cylinder``: toroidal and solid-cylinder sonar
  signatures differ); Propeller and Wall -> ``structure``; Chain is a
  hard-negative only (rigid links != filament mesh — never trained as
  net_like/ghost_net).
* KLSG is classification-only: manifest carries ``image_class`` per image,
  no YOLO export.

Usage::

    python scripts/convert_datasets.py [--dataset ai4shipwrecks|sctd|klsg|all]
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "training"

CLASSES = ["wreck", "debris", "structure", "tire"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}

MIN_BOX_AREA = 64  # px^2; skip mask specks
MIN_COMP_AREA = 48


def write_manifest(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_yolo(out_dir: Path, entries: list[dict]) -> None:
    """entries: [{image: abs_path, boxes: [{x,y,w,h,class}]}] -> yolo export."""
    ydir = out_dir / "yolo"
    (ydir / "images").mkdir(parents=True, exist_ok=True)
    (ydir / "labels").mkdir(parents=True, exist_ok=True)
    (ydir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    import shutil

    for i, e in enumerate(entries):
        src = Path(e["image"])
        dst = ydir / "images" / f"{src.parent.name}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        h, w = e["height"], e["width"]
        lines = []
        for b in e["boxes"]:
            cid = CLS_ID[b["class"]]
            cx, cy = (b["x"] + b["w"] / 2) / w, (b["y"] + b["h"] / 2) / h
            nw, nh = b["w"] / w, b["h"] / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (ydir / "labels" / (dst.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------- AI4Shipwrecks
def convert_ai4shipwrecks() -> None:
    src = RAW / "ai4shipwrecks" / "AI4Shipwrecks"
    if not src.exists():
        print("SKIP ai4shipwrecks: not downloaded")
        return
    out_dir = OUT / "ai4shipwrecks"
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    attribution = (
        "AI4Shipwrecks - Sheppard, Sethuraman, Bagoren, Pinnow, Anderson, Havens, Skinner (2024), "
        "NOAA Thunder Bay National Marine Sanctuary. DOI 10.7302/dmf4-x492. CC-BY 4.0."
    )
    entries, gt, mask_count = [], [], 0
    for split in ("train", "test"):
        for img_p in sorted((src / split / "images").glob("*.png")):
            lbl_p = src / split / "labels" / img_p.name
            if not lbl_p.exists():
                continue
            mask = cv2.imread(str(lbl_p), cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape[:2]
            boxes = []
            binm = (mask > 0).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
            inst = np.zeros_like(binm, dtype=np.uint8)
            for k in range(1, n):
                x, y, bw, bh, area = stats[k]
                if area < MIN_COMP_AREA:
                    continue
                inst[binm == k] = len(boxes) + 1
                if bw * bh >= MIN_BOX_AREA:
                    boxes.append(
                        {
                            "x": int(x),
                            "y": int(y),
                            "w": int(bw),
                            "h": int(bh),
                            "class": "wreck",
                            "raw_class": "shipwreck",
                        }
                    )
            if boxes:
                cv2.imwrite(str(out_dir / "masks" / img_p.name), inst)
                mask_count += 1
            entries.append({"image": str(img_p), "height": int(h), "width": int(w), "boxes": boxes})
            gt.append({"image": f"{split}/images/{img_p.name}", "boxes": boxes})
    terrain = [p.name for p in sorted((src / "extras" / "terrain" / "images").glob("*.png"))]
    manifest = {
        "id": "ai4shipwrecks",
        "name": "AI4Shipwrecks (real SSS)",
        "description": (
            "286 real side-scan sonar waterfall images of 24 shipwreck sites (NOAA Thunder Bay) with expert "
            "binary segmentation labels; train/test split by wreck site. Derived per-instance boxes from masks; "
            f"{len(terrain)} terrain-only negatives in extras/terrain."
        ),
        "synthetic": False,
        "attribution": attribution,
        "license_note": "CC-BY 4.0 - attribution required; safe for demo and training.",
        "tags": ["real", "sss", "segmentation-gt", "detection-gt", "wreck", "site-split"],
        "survey_meta": {"name": "AI4Shipwrecks", "sonar_type": "sss", "preprocess_preset": "light"},
        "splits": {"train": "train/images", "test": "test/images"},
        "images": [
            f"{s}/images/{p.name}" for s in ("train", "test") for p in sorted((src / s / "images").glob("*.png"))
        ],
        "ground_truth": gt,
        "terrain_negatives": terrain,
    }
    write_manifest(out_dir, manifest)
    write_yolo(out_dir, entries)
    nboxes = sum(len(e["boxes"]) for e in entries)
    print(f"ai4shipwrecks: {len(entries)} images, {nboxes} wreck boxes, {mask_count} seg masks -> {out_dir}")


# ------------------------------------------------------------------------ SCTD
# Semantic review: SCTD "aircraft" = sunken aircraft wreck sites (same
# sunken-vessel semantic as ship) -> wreck. "human" = a diver, NOT marine
# debris — human boxes are excluded from detector supervision entirely
# (v3 review). raw_class is always preserved for traceability.
SCTD_MAP = {"ship": "wreck", "aircraft": "wreck", "human": "_EXCLUDE_"}


def convert_sctd() -> None:
    src = RAW / "sctd" / "SCTD"
    if not src.exists():
        print("SKIP sctd: not downloaded")
        return
    out_dir = OUT / "sctd"
    entries, gt = [], []
    from PIL import Image

    for xml_p in sorted((src / "Annotations").glob("*.xml")):
        stem = xml_p.stem
        img_p = next((p for p in (src / "JPEGImages").glob(f"{stem}.*")), None)
        if img_p is None:
            continue
        with Image.open(img_p) as im:
            w, h = im.size
        boxes = []
        for obj in ET.parse(xml_p).getroot().findall("object"):
            raw = obj.findtext("name", "ship")
            if SCTD_MAP.get(raw) == "_EXCLUDE_":
                continue  # human: excluded from supervision (not debris)
            bb = obj.find("bndbox")
            xmin, ymin = float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0"))
            xmax, ymax = float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0"))
            boxes.append(
                {
                    "x": int(xmin),
                    "y": int(ymin),
                    "w": int(xmax - xmin),
                    "h": int(ymax - ymin),
                    "class": SCTD_MAP.get(raw, "debris"),
                    "raw_class": raw,
                }
            )
        entries.append({"image": str(img_p), "height": int(h), "width": int(w), "boxes": boxes})
        gt.append({"image": f"JPEGImages/{img_p.name}", "boxes": boxes})
    attribution = (
        "SCTD 1.0 - Sonar Common Target Detection Dataset, Ning et al. / Zhang et al. (IEEE TGRS), "
        "github.com/MingqiangNing/SCTD. Academic use; mixed side-scan / forward-looking / SAS imagery."
    )
    manifest = {
        "id": "sctd",
        "name": "SCTD 1.0 (mixed sonar)",
        "description": (
            "357 high-resolution sonar images (SSS/FLS/SAS mix) with 363 VOC boxes: ship 271, aircraft 57, "
            "human 35. ship/aircraft->wreck (328 boxes); human EXCLUDED from detector supervision "
            "(raw_class preserved, not marine debris)."
        ),
        "synthetic": False,
        "attribution": attribution,
        "license_note": "No explicit license in repo - academic/internal use only; verify before public release.",
        "tags": ["real", "mixed-sonar", "detection-gt"],
        "survey_meta": {"name": "SCTD 1.0", "sonar_type": "mixed", "preprocess_preset": "light"},
        "images": [f"JPEGImages/{Path(e['image']).name}" for e in entries],
        "ground_truth": gt,
    }
    write_manifest(out_dir, manifest)
    write_yolo(out_dir, entries)
    classes = Counter(b["class"] for e in entries for b in e["boxes"])
    print(f"sctd: {len(entries)} images, {sum(classes.values())} boxes {dict(classes)} -> {out_dir}")


# ------------------------------------------------------------------------ KLSG
def convert_klsg() -> None:
    src = RAW / "klsg"
    if not src.exists():
        print("SKIP klsg: not downloaded")
        return
    out_dir = OUT / "klsg"
    images, gt = [], []
    counts = Counter()
    for label, folder in (("ship", "ship-real"), ("airplane", "plane-real")):
        for img_p in sorted((src / folder).glob("*")):
            if img_p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            rel = f"{folder}/{img_p.name}"
            images.append(rel)
            gt.append({"image": rel, "boxes": [], "image_class": label})
            counts[label] += 1
    attribution = (
        "SeabedObjects-KLSG - open side-scan sonar dataset (385 ship, 62 airplane images), "
        "github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset. Academic use."
    )
    manifest = {
        "id": "klsg",
        "name": "SeabedObjects-KLSG (real SSS, classification)",
        "description": (
            "447 real side-scan sonar whole-image classification samples: 385 ship + 62 airplane. "
            "No boxes - used for natural-vs-artificial classifier training and hard-negative mining context."
        ),
        "synthetic": False,
        "attribution": attribution,
        "license_note": "Open for academic use per repo README - verify before public release.",
        "tags": ["real", "sss", "classification-only"],
        "survey_meta": {"name": "SeabedObjects-KLSG", "sonar_type": "sss", "preprocess_preset": "light"},
        "images": images,
        "ground_truth": gt,
    }
    write_manifest(out_dir, manifest)
    print(f"klsg: {len(images)} images {dict(counts)} (classification-only, no YOLO) -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["ai4shipwrecks", "sctd", "klsg", "all"])
    args = ap.parse_args()
    if args.dataset in ("ai4shipwrecks", "all"):
        convert_ai4shipwrecks()
    if args.dataset in ("sctd", "all"):
        convert_sctd()
    if args.dataset in ("klsg", "all"):
        convert_klsg()
    sys.exit(0)


if __name__ == "__main__":
    main()
