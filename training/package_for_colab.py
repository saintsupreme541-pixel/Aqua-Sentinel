"""Stage 1 — package every local dataset into one Colab-uploadable bundle.

Reads the converted manifests under ``data/training/`` (produced by
``backend/scripts/convert_datasets.py``) plus the MD-FLS watertank source and
writes ``aqua_train_bundle.zip`` containing::

    bundle/
      detect/{ai4shipwrecks,sctd,md_fls}/images + labels (YOLO .txt)
      detect/{...}/site_map.json          # image -> site key (leak-safe splits)
      seg/ai4shipwrecks/{images,masks}/
      seg/ai4shipwrecks/site_map.json
      cls/{artificial,natural}/*.png      # 64x64-ready crops (unresized)
      cls/crops_index.csv                 # crop -> source, class, site
      config.yaml, requirements-colab.txt
      DATASOURCES.md                      # full attribution

Site keys (used for leak-safe splits):
* AI4Shipwrecks: wreck name (e.g. ``EB_Allen`` from ``EB_Allen_08.png``) —
  the dataset's own site split is preserved.
* SCTD: image stem (independent captures; no sequence structure).
* MD-FLS: image stem (independent tank frames).

Usage (from repo root)::

    backend/.venv/Scripts/python.exe training/package_for_colab.py \
        [--out aqua_train_bundle.zip]
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "training"
RAW = ROOT / "data" / "raw"
SAMPLES = ROOT / "data" / "samples"
MD_FLS_SRC = ROOT / "marine-debris-fls-datasets-master" / "md_fls_dataset" / "data" / "watertank-segmentation"
BUNDLE = "bundle"

# Supervised taxonomy — ONLY classes with genuine labelled examples in the
# source data (semantic review v3 in docs/dataset-cards.md). Classes with no
# real training data (ghost_net, pipe, cylinder) are deliberately ABSENT: the
# detector must not claim them; unsupported objects route through the
# Unknown-Anomaly pathway at inference time instead.
#
# v3: Tire -> `tire`, NOT `cylinder`. A tire is a toroidal/ring-shaped
# manufactured object; toroidal and solid-cylinder sonar signatures differ in
# shadow shape and highlight topology. MD-FLS contains no solid cylinders,
# so `cylinder` = 0 verified training examples and is not trained.
CLASSES = ["wreck", "debris", "structure", "tire"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}

# MD-FLS raw VOC class -> taxonomy (same alias table as the backend importer)
MD_FLS_ALIAS = {
    "Bottle": "debris",
    "Can": "debris",
    # Crops/provenance only — detection SKIPS Chain (HARD_NEGATIVE) before this
    # lookup, so no chain box ever becomes a supervised target.
    "Chain": "structure",
    "Drink-carton": "debris",
    "Hook": "debris",
    # Propeller is intact man-made hard structure, not a sunken-vessel wreck
    # semantic — kept as its own labelled class (structure).
    "Propeller": "structure",
    "Shampoo-bottle": "debris",
    "Standing-bottle": "debris",
    # v3: tire keeps its own supervised class — toroidal ≠ solid cylinder.
    "Tire": "tire",
    "Valve": "debris",
    "Wall": "structure",
}

# Semantic review v2: "Chain" is NOT net_like — a chain is rigid metal links,
# a ghost net is flexible filament mesh (different geometry, texture, echo).
# Chain boxes are EXCLUDED from detector supervision (implicit background =
# hard negative); chain crops remain in the nat-vs-artificial set.
HARD_NEGATIVE = {"Chain"}


def site_key_ai4(stem: str) -> str:
    return re.split(r"_\d+$", stem)[0] or stem


def load_manifest_boxes(dset: str) -> dict:
    return json.loads((TRAIN / dset / "manifest.json").read_text(encoding="utf-8"))


def write_yolo_split(zf: zipfile.ZipFile, dset: str, items: list[dict], split: str) -> None:
    for it in items:
        src = Path(it["image"])
        arc_img = f"{BUNDLE}/detect/{dset}/{split}/images/{src.name}"
        arc_lbl = f"{BUNDLE}/detect/{dset}/{split}/labels/{src.stem}.txt"
        zf.write(src, arc_img)
        lines = []
        w, h = it["width"], it["height"]
        for b in it["boxes"]:
            cid = CLS_ID[b["class"]]
            cx, cy = (b["x"] + b["w"] / 2) / w, (b["y"] + b["h"] / 2) / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {b['w'] / w:.6f} {b['h'] / h:.6f}")
        zf.writestr(arc_lbl, "\n".join(lines) + ("\n" if lines else ""))


def write_classes_txt(zf: zipfile.ZipFile, dset: str) -> None:
    zf.writestr(f"{BUNDLE}/detect/{dset}/classes.txt", "\n".join(CLASSES) + "\n")


def package_ai4_detect(zf: zipfile.ZipFile) -> dict:
    write_classes_txt(zf, "ai4shipwrecks")
    m = load_manifest_boxes("ai4shipwrecks")
    raw_root = RAW / "ai4shipwrecks" / "AI4Shipwrecks"
    items = []
    for g in m["ground_truth"]:
        img_p = raw_root / g["image"]
        if not img_p.exists():
            continue

        from PIL import Image

        with Image.open(img_p) as im:
            w, h = im.size
        split = g["image"].split("/")[0]
        items.append({"image": str(img_p), "width": w, "height": h, "boxes": g["boxes"], "split": split})
    site_map = {Path(it["image"]).stem: site_key_ai4(Path(it["image"]).stem) for it in items}
    zf.writestr(f"{BUNDLE}/detect/ai4shipwrecks/site_map.json", json.dumps(site_map, indent=2))
    for split in ("train", "test"):
        write_yolo_split(zf, "ai4shipwrecks", [it for it in items if it["split"] == split], split)
    return {"images": len(items), "boxes": sum(len(i["boxes"]) for i in items), "sites": len(set(site_map.values()))}


def package_sctd_detect(zf: zipfile.ZipFile) -> dict:
    write_classes_txt(zf, "sctd")
    m = load_manifest_boxes("sctd")
    raw_root = RAW / "sctd" / "SCTD"
    items = []
    for g in m["ground_truth"]:
        img_p = raw_root / g["image"]
        if not img_p.exists():
            continue
        from PIL import Image

        with Image.open(img_p) as im:
            w, h = im.size
        items.append({"image": str(img_p), "width": w, "height": h, "boxes": g["boxes"]})
    site_map = {Path(it["image"]).stem: Path(it["image"]).stem for it in items}
    zf.writestr(f"{BUNDLE}/detect/sctd/site_map.json", json.dumps(site_map, indent=2))
    write_yolo_split(zf, "sctd", items, "train")
    return {"images": len(items), "boxes": sum(len(i["boxes"]) for i in items)}


MD_FLS_SPLIT = re.compile(r"^(.*?)[-_]")


def package_mdfls_detect(zf: zipfile.ZipFile) -> dict:
    """All 1,868 MD-FLS watertank images with VOC boxes (incl. Wall)."""
    import xml.etree.ElementTree as ET

    if not MD_FLS_SRC.exists():
        print("SKIP md_fls: source clone missing")
        return {"images": 0}
    write_classes_txt(zf, "md_fls")
    items = []
    for xml_p in sorted((MD_FLS_SRC / "BoxAnnotations").glob("*.xml")):
        img_p = MD_FLS_SRC / "Images" / f"{xml_p.stem}.png"
        if not img_p.exists():
            continue
        boxes = []
        for obj in ET.parse(xml_p).getroot().findall("object"):
            raw = obj.findtext("name", "debris")
            if raw in HARD_NEGATIVE:
                continue  # chain = implicit background for the detector
            bb = obj.find("bndbox")
            x, y = float(bb.findtext("x", "0")), float(bb.findtext("y", "0"))
            w, h = float(bb.findtext("w", "0")), float(bb.findtext("h", "0"))
            boxes.append({"x": x, "y": y, "w": w, "h": h, "class": MD_FLS_ALIAS.get(raw, "debris"), "raw_class": raw})
        from PIL import Image

        with Image.open(img_p) as im:
            img_w, img_h = im.size  # read real size — never hard-code orientation
        items.append({"image": str(img_p), "width": img_w, "height": img_h, "boxes": boxes})
    site_map = {Path(it["image"]).stem: Path(it["image"]).stem for it in items}
    zf.writestr(f"{BUNDLE}/detect/md_fls/site_map.json", json.dumps(site_map, indent=2))
    write_yolo_split(zf, "md_fls", items, "train")
    return {"images": len(items), "boxes": sum(len(i["boxes"]) for i in items)}


def package_ai4_seg(zf: zipfile.ZipFile) -> dict:
    seg_masks = TRAIN / "ai4shipwrecks" / "masks"
    raw_root = RAW / "ai4shipwrecks" / "AI4Shipwrecks"
    n = 0
    site_map = {}
    for msk_p in sorted(seg_masks.glob("*.png")):
        for split in ("train", "test"):
            if (raw_root / split / "images" / msk_p.name).exists():
                img_p = raw_root / split / "images" / msk_p.name
                break
        else:
            continue
        zf.write(img_p, f"{BUNDLE}/seg/ai4shipwrecks/images/{msk_p.name}")
        zf.write(msk_p, f"{BUNDLE}/seg/ai4shipwrecks/masks/{msk_p.name}")
        site_map[msk_p.stem] = site_key_ai4(msk_p.stem)
        n += 1
    zf.writestr(f"{BUNDLE}/seg/ai4shipwrecks/site_map.json", json.dumps(site_map, indent=2))
    return {"pairs": n, "sites": len(set(site_map.values()))}


def crop_boxes(img: np.ndarray, boxes: list[dict], pad: float = 0.08) -> list[np.ndarray]:
    h, w = img.shape[:2]
    out = []
    for b in boxes:
        x0 = max(0, int(b["x"] - pad * b["w"]))
        y0 = max(0, int(b["y"] - pad * b["h"]))
        x1 = min(w, int(b["x"] + b["w"] * (1 + pad)))
        y1 = min(h, int(b["y"] + b["h"] * (1 + pad)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        out.append(img[y0:y1, x0:x1])
    return out


def package_cls(zf: zipfile.ZipFile) -> dict:
    """Crop-level natural-vs-artificial set with provenance CSV."""
    import csv
    import io

    rows = [("crop", "label", "source", "site", "raw_class")]
    counts = Counter()
    idx = 0

    def add(img: np.ndarray, label: str, source: str, site: str, raw: str) -> None:
        nonlocal idx
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return
        name = f"{label}_{source}_{idx:05d}.png"
        zf.writestr(f"{BUNDLE}/cls/{label}/{name}", buf.tobytes())
        rows.append((name, label, source, site, raw))
        counts[label] += 1
        idx += 1

    # AI4Shipwrecks wreck crops (artificial) + terrain crops (natural context)
    ai4 = RAW / "ai4shipwrecks" / "AI4Shipwrecks"
    m = load_manifest_boxes("ai4shipwrecks")
    for g in m["ground_truth"]:
        img_p = ai4 / g["image"]
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        site = site_key_ai4(Path(g["image"]).stem)
        for c in crop_boxes(img, g["boxes"]):
            add(c, "artificial", "ai4shipwrecks", site, "shipwreck")
    for t in sorted((ai4 / "extras" / "terrain" / "images").glob("*.png")):
        img = cv2.imread(str(t), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for c in crop_boxes(img, [{"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[0]}]):
            add(c, "natural", "ai4_terrain", "terrain", "terrain")

    # SCTD crops
    sctd = RAW / "sctd" / "SCTD"
    ms = load_manifest_boxes("sctd")
    for g in ms["ground_truth"]:
        img_p = sctd / g["image"]
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        site = Path(g["image"]).stem
        for b in g["boxes"]:
            for c in crop_boxes(img, [b]):
                add(c, "artificial", "sctd", site, b.get("raw_class", b["class"]))

    # MD-FLS crops: debris/cylinder/structure targets = artificial; Wall = natural-context.
    # Chain crops ARE included here (artificial side) — the hard-negative
    # exclusion applies to detector supervision only.
    import xml.etree.ElementTree as ET

    if MD_FLS_SRC.exists():
        for xml_p in sorted((MD_FLS_SRC / "BoxAnnotations").glob("*.xml")):
            img_p = MD_FLS_SRC / "Images" / f"{xml_p.stem}.png"
            if not img_p.exists():
                continue
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            boxes = []
            for obj in ET.parse(xml_p).getroot().findall("object"):
                raw = obj.findtext("name", "debris")
                bb = obj.find("bndbox")
                boxes.append(
                    {
                        "x": float(bb.findtext("x", "0")),
                        "y": float(bb.findtext("y", "0")),
                        "w": float(bb.findtext("w", "0")),
                        "h": float(bb.findtext("h", "0")),
                        "raw_class": raw,
                        "class": MD_FLS_ALIAS.get(raw, "debris"),
                    }
                )
            # v3: artificial = everything except Wall. Human crops are already
            # excluded upstream (human never enters MD-FLS anyway); chain
            # crops remain here by review decision (classifier hard negative).
            natural = [b for b in boxes if b["raw_class"] == "Wall"]
            artificial = [b for b in boxes if b["raw_class"] != "Wall"]
            for c in crop_boxes(img, artificial):
                add(c, "artificial", "md_fls", xml_p.stem, "mixed")
            for c in crop_boxes(img, natural):
                add(c, "natural", "md_fls_wall", xml_p.stem, "Wall")

    # KLSG whole images downscaled (artificial)
    klsg = RAW / "klsg"
    for folder, raw in (("ship-real", "ship"), ("plane-real", "airplane")):
        for p in sorted((klsg / folder).glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            add(img, "artificial", "klsg", folder, raw)

    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    zf.writestr(f"{BUNDLE}/cls/crops_index.csv", buf.getvalue())
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "aqua_train_bundle.zip")
    args = ap.parse_args()

    stats: dict = {}
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in ("config.yaml", "requirements-colab.txt"):
            zf.write(Path(__file__).parent / f, f"{BUNDLE}/{f}")
        # The notebook calls training/*.py from the extracted bundle root —
        # ship the scripts inside the zip so Colab needs no git clone.
        for script in sorted(Path(__file__).parent.glob("*.py")):
            zf.write(script, f"training/{script.name}")
        # pinned deps the notebook installs from (single source of truth)
        req = Path(__file__).parent / "requirements-colab.txt"
        if req.exists():
            zf.write(req, "training/requirements-colab.txt")
        stats["detect_ai4"] = package_ai4_detect(zf)
        stats["detect_sctd"] = package_sctd_detect(zf)
        stats["detect_mdfls"] = package_mdfls_detect(zf)
        stats["seg_ai4"] = package_ai4_seg(zf)
        stats["cls_counts"] = package_cls(zf)
        stats["detect_classes"] = CLASSES

        zf.writestr(
            f"{BUNDLE}/DATASOURCES.md",
            "# Data sources\n\n"
            "- AI4Shipwrecks: Sheppard et al. 2024, NOAA Thunder Bay, DOI 10.7302/dmf4-x492, CC-BY 4.0.\n"
            "- SCTD 1.0: github.com/MingqiangNing/SCTD, no explicit license (academic/internal use).\n"
            "- MD-FLS watertank: Valdenegro-Toro et al., OCEANS 2025, doi:10.1109/OCEANS58557.2025.11104623; "
            "no explicit license (academic/internal use).\n"
            "- SeabedObjects-KLSG: github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset, academic use.\n",
        )
    print(json.dumps(stats, indent=2))
    print(f"bundle: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
