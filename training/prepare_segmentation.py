"""Stage 2b — site-level split + dataset/tile loader for U-Net segmentation.

AI4Shipwrecks masks: 0 = background, 1..N = wreck instances. We binarize
(>0) for the single-class wreck segmentation task (matches the app's ONNX
segmenter: full-image single-channel logits, resize to input size).

Split: by wreck site (site_map.json), 85/15 — identical policy to the
app-side split_dataset.py philosophy (no frames of the same site across
splits). Terrain images (all-negative masks) go to TRAIN only as extra
background supervision.

Usage:  python prepare_segmentation.py            # build splits
        python prepare_segmentation.py --emit-filelists
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

B = Path("/content/aqua_train/bundle")
SEG = B / "seg" / "ai4shipwrecks"


def site_split(val_frac: float = 0.15, seed: int = 20260905) -> dict:
    smap = json.loads((SEG / "site_map.json").read_text(encoding="utf-8"))
    sites = sorted(set(smap.values()))
    rng = random.Random(seed)
    rng.shuffle(sites)
    n_val = max(1, round(len(sites) * val_frac))
    val_sites = set(sites[:n_val])
    tr, va = [], []
    for stem, site in sorted(smap.items()):
        (va if site in val_sites else tr).append(stem)
    return {"train": tr, "val": va, "val_sites": sorted(val_sites)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-filelists", action="store_true")
    args = ap.parse_args()

    split = site_split()
    OUT_YAML = SEG / "seg.yaml"
    (SEG / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    yaml_body = {
        "task": "binary-segmentation",
        "class": "wreck",
        "images": str(SEG / "images"),
        "masks": str(SEG / "masks"),
        "train": split["train"],
        "val": split["val"],
        "note": "site-level split; binarized instance masks (>0); terrain negatives in train only",
    }
    OUT_YAML.write_text(yaml.safe_dump(yaml_body, sort_keys=False), encoding="utf-8")
    print(f"seg split: train={len(split['train'])} val={len(split['val'])} val_sites={split['val_sites']}")
    if args.emit_filelists:
        for name, stems in (("train", split["train"]), ("val", split["val"])):
            (SEG / f"{name}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
