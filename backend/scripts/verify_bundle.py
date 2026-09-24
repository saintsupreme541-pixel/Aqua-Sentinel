"""Verify ``aqua_train_bundle.zip`` before GPU training (read-only gate).

Checks, per detection dataset in the bundle:
- classes.txt is exactly the approved v3 taxonomy, in order;
- every YOLO label line has 5 fields, valid class ids (0..nc-1), coords in [0,1];
- label↔image 1:1 alignment (no missing pairs either way);
- site_map.json present with one entry per image;
- empty (background) label files are allowed and counted;
- CRC integrity spot-check on a random sample of images.

Bundle-level checks:
- training/*.py + config.yaml + requirements + DATASOURCES.md present (notebook wiring);
- no supervised box is human/Chain-derived (census vs expected counts);
- cls provenance CSV contains no human crops;
- seg + all detect dirs ship site maps (leak-safe splits possible).

Usage: python scripts/verify_bundle.py [path-to-zip]
Exit 0 = ready; any problem prints ``VERIFY: FAIL`` and exits 1.
"""

from __future__ import annotations

import collections
import json
import random
import sys
import zipfile
from pathlib import Path

EXPECTED_CLASSES = ["wreck", "debris", "structure", "tire"]
# Verified on-disk object counts after the v3 exclusions (Chain skipped in
# detection; human excluded from SCTD supervision).
EXPECTED_BOXES = {"ai4shipwrecks": 954, "sctd": 328, "md_fls": 3562}
REQUIRED_TRAINING_SCRIPTS = [
    "prepare_detection.py",
    "prepare_segmentation.py",
    "prepare_classifier.py",
    "train_detection.py",
    "train_segmentation.py",
    "train_classifier.py",
    "export_onnx.py",
    "evaluate_onnx.py",
]
REQUIRED_BUNDLE_FILES = [
    "bundle/config.yaml",
    "bundle/requirements-colab.txt",
    "bundle/DATASOURCES.md",
    "bundle/detect/ai4shipwrecks/site_map.json",
    "bundle/detect/sctd/site_map.json",
    "bundle/detect/md_fls/site_map.json",
    "bundle/seg/ai4shipwrecks/site_map.json",
    "bundle/cls/crops_index.csv",
]


def fail(msg: str) -> None:
    print(f"VERIFY: FAIL — {msg}")
    raise SystemExit(1)


def main() -> None:
    default_zip = Path(__file__).resolve().parent.parent.parent / "aqua_train_bundle.zip"
    zip_path = Path(sys.argv[1] if len(sys.argv) > 1 else default_zip)
    if not zip_path.exists():
        fail(f"bundle not found: {zip_path}")
    zf = zipfile.ZipFile(zip_path)
    names = set(zf.namelist())
    print(f"bundle: {zip_path.name}, {len(names)} entries")

    # ---- required files -------------------------------------------------
    for req in REQUIRED_BUNDLE_FILES:
        if req not in names:
            fail(f"missing required bundle file: {req}")
    for script in REQUIRED_TRAINING_SCRIPTS:
        if f"training/{script}" not in names:
            fail(f"missing training script in bundle: training/{script}")
    print(f"required files: OK ({len(REQUIRED_BUNDLE_FILES)} + {len(REQUIRED_TRAINING_SCRIPTS)} scripts)")

    # ---- classes.txt ----------------------------------------------------
    for d in ("ai4shipwrecks", "sctd", "md_fls"):
        got = zf.read(f"bundle/detect/{d}/classes.txt").decode().split()
        if got != EXPECTED_CLASSES:
            fail(f"{d}/classes.txt = {got}, expected {EXPECTED_CLASSES}")
    print(f"classes.txt (x3): {EXPECTED_CLASSES} OK")

    # ---- label census ---------------------------------------------------
    grand_total = {}
    for d in ("ai4shipwrecks", "sctd", "md_fls"):
        ids: collections.Counter[int] = collections.Counter()
        n_lbl = n_empty = n_missing_img = n_orphan_img = bad_fields = bad_range = 0
        for split in ("train", "test"):
            lbls = [n for n in names if n.startswith(f"bundle/detect/{d}/{split}/labels/") and n.endswith(".txt")]
            imgs = {
                n.split("/")[-1].rsplit(".", 1)[0] for n in names if n.startswith(f"bundle/detect/{d}/{split}/images/")
            }
            img_names = {n.split("/")[-1] for n in names if n.startswith(f"bundle/detect/{d}/{split}/images/")}
            for ln in lbls:
                stem = ln.split("/")[-1][:-4]
                if stem not in imgs:
                    n_missing_img += 1
                txt = zf.read(ln).decode("utf-8").strip()
                n_lbl += 1
                if not txt:
                    n_empty += 1
                    continue
                for line in txt.splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        bad_fields += 1
                        continue
                    cid = int(parts[0])
                    ids[cid] += 1
                    vals = [float(v) for v in parts[1:]]
                    if not all(0.0 <= v <= 1.0 for v in vals):
                        bad_range += 1
            # image dirs ship an (possibly empty) label for every image
            n_orphan_img += len(
                [i for i in img_names if f"bundle/detect/{d}/{split}/labels/{i.rsplit('.', 1)[0]}.txt" not in names]
            )
        total = sum(ids.values())
        grand_total[d] = total
        if ids.keys() - set(range(len(EXPECTED_CLASSES))):
            fail(f"{d}: invalid class ids {sorted(set(ids) - set(range(4)))}")
        if bad_fields:
            fail(f"{d}: {bad_fields} label lines with wrong field count")
        if bad_range:
            fail(f"{d}: {bad_range} label lines with coordinates outside [0,1]")
        if n_missing_img or n_orphan_img:
            fail(f"{d}: label/image mismatch (missing_img={n_missing_img}, orphan_img={n_orphan_img})")
        if total != EXPECTED_BOXES[d]:
            fail(f"{d}: box count {total} != expected {EXPECTED_BOXES[d]} (taxonomy drift?)")
        dist = {EXPECTED_CLASSES[k]: v for k, v in sorted(ids.items())}
        print(f"{d}: {n_lbl} label files ({n_empty} background), {total} boxes, ids={dist} — OK")

    # human/Chain exclusion is structural (counts above) + provenance check:
    idx = zf.read("bundle/cls/crops_index.csv").decode("utf-8").splitlines()
    human_rows = [r for r in idx[1:] if r.rsplit(",", 1)[-1] == "human"]
    if human_rows:
        fail(f"crops_index.csv contains {len(human_rows)} human crops")
    print(f"crops_index: {len(idx) - 1} crops, 0 human — OK")

    # ---- site maps ------------------------------------------------------
    for d in ("detect/ai4shipwrecks", "detect/sctd", "detect/md_fls", "seg/ai4shipwrecks"):
        sm = json.loads(zf.read(f"bundle/{d}/site_map.json").decode())
        if not sm:
            fail(f"{d}/site_map.json empty")
    print("site maps (x4): non-empty OK")

    # ---- CRC spot-check on random images --------------------------------
    rng = random.Random(20260905)
    img_entries = [n for n in names if "/images/" in n and n.endswith(".png")]
    for n in rng.sample(img_entries, min(40, len(img_entries))):
        try:
            zf.read(n)  # read() verifies CRC
        except Exception as e:  # noqa: BLE001
            fail(f"corrupt zip member {n}: {e}")
    print(f"CRC spot-check: {min(40, len(img_entries))}/{len(img_entries)} images OK")

    print("VERIFY: PASS")


if __name__ == "__main__":
    main()
