"""Survey-aware train/val/test splitter for labelled sonar imagery.

Adjacent sonar pings overlap, so *random image splits leak*: the same object
appears in neighbouring frames and inflates validation metrics.  This tool
groups images by a survey key — by default their stem prefix before the
first ``-``/``_`` (e.g. ``site42-ping017.png`` → group ``site42``), or the
parent folder when ``--by-folder`` is set — and splits whole groups.

Usage::

    python scripts/split_dataset.py <images_dir> \\
        --by-folder --val 0.2 --test 0.2 --seed 7 \\
        --out data/splits/watertank

Writes three manifest files (train.json / val.json / test.json) that list
the split image stems, plus a split_report.json with the group counts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

STEM_GROUP = re.compile(r"^(.*?)[-_]")


def group_key(path: Path, by_folder: bool) -> str:
    if by_folder:
        return str(path.parent)
    m = STEM_GROUP.match(path.stem)
    return m.group(1) if m else path.stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", type=Path)
    ap.add_argument(
        "--by-folder",
        action="store_true",
        help="treat each sub-folder as a survey (recommended for per-survey datasets)",
    )
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    args = ap.parse_args()

    if not args.images_dir.is_dir():
        raise SystemExit(f"not a directory: {args.images_dir}")
    images = sorted(
        p for p in args.images_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    if not images:
        raise SystemExit(f"no images found under {args.images_dir}")

    groups: dict[str, list[str]] = defaultdict(list)
    for p in images:
        groups[group_key(p, args.by_folder)].append(str(p))
    items = sorted(groups.items())

    if len(items) < 3:
        # single-survey folder (common for SSS/FLS dumps): no survey-level split
        # exists.  Fall back to a *chronological* split and say so loudly:
        # adjacent pings overlap, so metric optimism must be disclosed.
        print(
            f"! only {len(items)} survey group(s); chronological fallback split. "
            "Adjacent-ping overlap can leak across split boundaries - report metrics with this caveat."
        )
        items = [(str(i), [str(p)]) for i, p in enumerate(sorted(images))]

    rng = random.Random(args.seed)
    rng.shuffle(items)
    n = len(items)
    n_test = round(n * args.test)
    n_val = round(n * args.val)
    n_train = n - n_test - n_val
    splits = {"train": items[:n_train], "val": items[n_train : n_train + n_val], "test": items[n_train + n_val :]}

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"groups_total": n, "images_total": len(images), "by": "folder" if args.by_folder else "stem-prefix"}
    for name, group_items in splits.items():
        files = [f for _, files in group_items for f in files]
        (args.out / f"{name}.json").write_text(
            json.dumps({"split": name, "groups": len(group_items), "images": files}, indent=2), encoding="utf-8"
        )
        report[name] = {"groups": len(group_items), "images": len(files)}
        print(f"{name:5s}: {len(group_items):3d} groups / {len(files):4d} images  -> {args.out / name}.json")
    (args.out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if n == len(images):
        print("\nChronological fallback used: whole surveys could not be separated.")
    else:
        print("\nKept whole surveys together - no ping-level leakage between splits.")


if __name__ == "__main__":
    main()
