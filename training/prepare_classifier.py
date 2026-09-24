"""Stage 2c — build the 64×64 natural-vs-artificial crop dataset.

Reads  bundle/cls/{artificial,natural}/*.png + crops_index.csv
Writes bundle/cls/split/{train,val}/{artificial,natural}/*.png  (64×64)

Split is site-level: crops from the same wreck site / source image never
straddle train and val (adjacent crops of one image are near-duplicates).
MD-FLS and KLSG have per-capture singleton sites -> their whole source image
is one site; the split is still group-wise, never per-crop random.

Also emits class weights for the imbalance (natural is the minority).
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

B = Path("/content/aqua_train/bundle")
CLS = B / "cls"
OUT = CLS / "split"
SIZE = 64


def site_of(source: str, site: str) -> str:
    if source == "ai4shipwrecks":
        return "ai4_" + site
    if source == "md_fls_wall":
        return "mdfls_" + site
    if source == "md_fls":
        return "mdfls_" + site
    if source == "sctd":
        return "sctd_" + site
    return source + "_" + site  # klsg folders


def main() -> None:
    rows = list(csv.DictReader((CLS / "crops_index.csv").read_text(encoding="utf-8").splitlines()))
    groups: dict[str, list[dict]] = defaultdict(list)
    # DictReader already skipped the header — iterating rows directly avoids
    # dropping the first real crop (a prior [1:] slice silently ate it).
    for r in rows:
        groups[site_of(r["source"], r["site"])].append(r)

    sites = sorted(groups)
    rng = random.Random(20260905)
    rng.shuffle(sites)
    n_val = max(1, round(len(sites) * 0.15))
    val_sites = set(sites[:n_val])

    counts = Counter()
    for site, items in groups.items():
        split = "val" if site in val_sites else "train"
        for it in items:
            src = CLS / it["label"] / it["crop"]
            dst = OUT / split / it["label"]
            dst.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst / it["crop"]), img)
            counts[(split, it["label"])] += 1

    n_tr_a, n_tr_n = counts[("train", "artificial")], counts[("train", "natural")]
    pos_weight = round(n_tr_n / max(n_tr_a, 1), 4) if n_tr_a else 1.0
    meta = {
        "val_sites": sorted(val_sites),
        "counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
        "pos_weight_for_bce": pos_weight,
        "note": "weight applied to the positive (artificial) class so loss matches imbalance",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
