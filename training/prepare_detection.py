"""Stage 2a — site-level train/val split for YOLO detection.

Group-by-site split: all frames from one wreck site / capture stay in one
split, so near-duplicate sonar views cannot leak from train into val.
MD-FLS has no site structure (single tank session) -> documented fallback:
**hold out a chronological tail** (file order) as val and flag it in the
report — val metrics on that split are optimistic and must be labeled as such.

Reads  bundle/detect/<dset>/{train|test}/... + site_map.json
Writes bundle/detect/<dset>/data.yaml + split_report.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

B = Path("/content/aqua_train/bundle")


def split_site_aware(files: list[str], site_map: dict, val_frac: float, seed: int) -> tuple[set[str], set[str], dict]:
    sites = sorted({site_map.get(Path(f).stem, Path(f).stem) for f in files})
    single_image_sites = defaultdict(list)
    for s in sites:
        single_image_sites[s].append(s)
    # A "site" with exactly one image gives no leakage protection but still
    # counts as a group; MD-FLS (all-singleton sites) triggers the fallback.
    if len(sites) < 4 or len(sites) >= 0.9 * len(files):
        return (
            set(),
            set(files),
            {"mode": "chronological-tail-fallback", "reason": "no multi-image site structure", "sites": len(sites)},
        )
    rng = random.Random(seed)
    rng.shuffle(sites)
    n_val = max(1, round(len(sites) * val_frac))
    val_sites = set(sites[:n_val])
    return set(sites[n_val:]), val_sites, {"mode": "site-level", "sites_total": len(sites), "val_sites": n_val}


def relink(dset_dir: Path, src_split: str, move_stems: set[str], dst_split: str) -> int:
    moved = 0
    (dset_dir / dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dset_dir / dst_split / "labels").mkdir(parents=True, exist_ok=True)
    for img in (dset_dir / src_split / "images").glob("*"):
        if img.stem in move_stems:
            dst = dset_dir / dst_split / "images" / img.name
            img.rename(dst)
            lbl = dset_dir / src_split / "labels" / (img.stem + ".txt")
            if lbl.exists():
                lbl.rename(dset_dir / dst_split / "labels" / lbl.name)
            moved += 1
    return moved


def main() -> None:
    det = B / "detect"
    reports = {}
    for dset_dir in sorted(p for p in det.iterdir() if p.is_dir()):
        smap_path = dset_dir / "site_map.json"
        if not smap_path.exists():
            continue
        smap = json.loads(smap_path.read_text(encoding="utf-8"))
        train_imgs = sorted((dset_dir / "train" / "images").glob("*"))
        files = [p.name for p in train_imgs]
        train_sites, val_sites, info = split_site_aware(files, smap, val_frac=0.18, seed=20260905)
        # site-level mode: keep only sites NOT in val_sites in train
        if info["mode"] == "site-level":
            keep = {Path(f).stem for f in files if smap.get(Path(f).stem, Path(f).stem) in train_sites}
            n = relink(dset_dir, "train", {Path(f).stem for f in files} - keep, "val")
        else:
            # chronological tail fallback: move last 15% by name order
            tail = {Path(f).stem for f in sorted(files)[int(len(files) * 0.85) :]}
            n = relink(dset_dir, "train", tail, "val")
            info["val_images"] = len(tail)
        reports[dset_dir.name] = info
        classes = (dset_dir / "classes.txt").read_text().split()
        data_yaml = {
            "path": str(dset_dir),
            "train": "train/images",
            "val": "val/images",
            "names": {i: c for i, c in enumerate(classes)},
        }
        (dset_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        print(f"{dset_dir.name}: {info} moved={n}")
    (B / "detect" / "split_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
