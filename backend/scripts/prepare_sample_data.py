"""Bundle a small, attributed subset of the Marine Debris FLS watertank
dataset into ``data/samples/md_fls_watertank`` with a manifest the backend
can import in one click.

Usage::

    python scripts/prepare_sample_data.py [--count 24] [--source <repo path>]

The subset is small and used only for the offline demo + evaluation
harness.  Full attribution lives in the manifest and docs/dataset-cards.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES = ROOT / "data" / "samples"

SRC_IMAGES = "marine-debris-fls-datasets-master/md_fls_dataset/data/watertank-segmentation"

# Semantic review v2 (docs/dataset-cards.md): Chain is rigid metal links, not
# filament mesh — mapped to `structure` (its own auditable class, excluded from
# detector supervision) rather than `net_like`. Propeller is intact man-made
# hard structure, not a sunken-vessel wreck semantic.
CLASS_ALIAS = {
    "Bottle": "debris",
    "Can": "debris",
    "Chain": "structure",
    "Drink-carton": "debris",
    "Hook": "debris",
    "Propeller": "structure",
    "Shampoo-bottle": "debris",
    "Standing-bottle": "debris",
    # v3: tire keeps its own supervised class — toroidal, never `cylinder`.
    "Tire": "tire",
    "Valve": "debris",
    "Wall": "structure",
}

# Detector hard-negative (semantic review v2): chain boxes are EXCLUDED from
# supervision — the object is left as implicit background so the detector
# learns to reject chain-like returns instead of fitting them to a net class.
HARD_NEGATIVE = {"Chain"}


def parse_boxes(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.getroot().findall("object"):
        name = obj.findtext("name", "debris")
        bb = obj.find("bndbox")
        x = float(bb.findtext("x", "0"))
        y = float(bb.findtext("y", "0"))
        w = float(bb.findtext("w", "0"))
        h = float(bb.findtext("h", "0"))
        if name in HARD_NEGATIVE:
            continue  # chain = implicit background for the detector
        boxes.append({"x": x, "y": y, "w": w, "h": h, "class": CLASS_ALIAS.get(name, "debris"), "raw_class": name})
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--source", type=Path, default=ROOT / SRC_IMAGES)
    args = ap.parse_args()

    src = args.source
    img_dir, msk_dir, box_dir = src / "Images", src / "Masks", src / "BoxAnnotations"
    if not (img_dir / "marine-debris-aris3k-0.png").exists():
        sys.exit(f"source dataset not found at {src}")

    images = sorted(img_dir.glob("*.png"))
    total = len(images)
    step = max(1, total // args.count)
    picked = [images[i] for i in range(0, total, step)][: args.count]

    out = SAMPLES / "md_fls_watertank"
    for sub in ("Images", "Masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    gt_list = []
    manifest_images = []
    for img_p in picked:
        stem = img_p.stem
        name = img_p.name
        (out / "Images" / name).write_bytes(img_p.read_bytes())
        msk_p = msk_dir / name
        if msk_p.exists():
            (out / "Masks" / name).write_bytes(msk_p.read_bytes())
        boxes = parse_boxes(box_dir / f"{stem}.xml")
        rel = f"Images/{name}"
        manifest_images.append(rel)
        gt_list.append({"image": rel, "boxes": boxes})

    attribution = (
        "Marine Debris Forward-Looking Sonar Datasets - M. Valdenegro-Toro et al., OCEANS 2025 Brest, "
        "doi:10.1109/OCEANS58557.2025.11104623 (arXiv:2503.22880). Labels by D. Singh et al."
    )
    license_note = (
        "Source repository (github.com/mvaldenegro/marine-debris-fls-datasets) declares no explicit license; "
        "bundled for internal demo/evaluation only - verify before any public release."
    )
    manifest = {
        "id": "md_fls_watertank",
        "name": "MD-FLS Watertank (demo subset)",
        "description": (
            "Real forward-looking sonar images of marine debris in a test tank (subset of 24 of 1,868 labelled images)."
        ),
        "synthetic": False,
        "attribution": attribution,
        "license_note": license_note,
        "tags": ["real", "fls", "segmentation-gt", "detection-gt"],
        "survey_meta": {"name": "MD-FLS Watertank (demo subset)", "sonar_type": "fls", "preprocess_preset": "light"},
        "images": manifest_images,
        "ground_truth": gt_list,
    }
    (SAMPLES / "md_fls_watertank.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- curated demo manifest (images whose confirmations match GT well) ----
    demo_names = {
        "marine-debris-aris3k-0.png",
        "marine-debris-aris3k-496.png",
        "marine-debris-aris3k-565.png",
        "marine-debris-aris3k-1067.png",
        "marine-debris-aris3k-1136.png",
        "marine-debris-aris3k-1344.png",
        "marine-debris-aris3k-1621.png",
        "marine-debris-aris3k-357.png",
    }
    demo_manifest = json.loads(json.dumps(manifest))
    demo_manifest["id"] = "md_fls_watertank_demo"
    demo_manifest["name"] = "MD-FLS Watertank (curated demo)"
    demo_manifest["description"] = (
        "Curated 8-image subset with clean, ground-truth-matching confirmations - one-click demo survey."
    )
    demo_manifest["images_root"] = "md_fls_watertank"
    demo_manifest["images"] = [rel for rel in manifest_images if Path(rel).name in demo_names]
    demo_manifest["ground_truth"] = [g for g in gt_list if Path(g["image"]).name in demo_names]
    (SAMPLES / "md_fls_watertank_demo.json").write_text(json.dumps(demo_manifest, indent=2), encoding="utf-8")

    readme = f"""# MD-FLS Watertank demo subset

Real Forward-Looking Sonar images (ARIS Explorer 3000) of marine debris in a test tank.
Source: Marine Debris Forward-Looking Sonar Datasets, Valdenegro-Toro et al., OCEANS 2025
(doi:10.1109/OCEANS58557.2025.11104623, arXiv:2503.22880).

- {len(picked)} of {total} labelled images (320×480, 8-bit grayscale)
- pixel masks in Masks/  ·  detection ground truth in the manifest (from the VOC XML labels)

The source repository declares no explicit license. This subset is bundled for the
internal demo and evaluation harness only — verify licensing before any public release.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {len(picked)} images to {out}")


if __name__ == "__main__":
    main()
