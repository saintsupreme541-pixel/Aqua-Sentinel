# Datasets for training (public SSS / marine-debris)

Training resources we identified during research, with realistic
fit-for-purpose notes. Verify licenses before redistribution.

## Detection (bounding boxes)

| Dataset | Content | Size | License / notes |
|---|---|---|---|
| **SSS Mine Detection** (Valdenegro-Toro) | SSS images of underwater mines | 1,170 labelled images | research use; cite |
| **AI4Shipwrecks** (García-Pineda et al.) | SSS shipwrecks, pixel-wise masks | 286 labelled images | cite; masks enable seg + split-by-wreck |
| **Marine_PULSE** | SSS pipes / mounds / platforms | 627 images | used for SSS pipe surveys |
| **SubPipe** | SSS pipeline imagery | ~10 k images | large; strong pipe baseline |
| **MD-FLS watertank** (this repo ships a subset) | FLS tank images of debris (nets, cans, chains, tires…) | 1,868 labelled + masks | source repo declares **no explicit license** → internal demo only |

## Segmentation / seafloor classes (hard negatives!)

| Dataset | Content | Why it matters |
|---|---|---|
| **BenthiCat** | ~37 k labelled SSS/MBES tiles, 26 seabed classes | rocks, ripples, texture classes are the *false-positive generators*; ideal for natural-vs-artificial training and OOD baselines |
| **AI4Shipwrecks masks** | wreck pixel masks | segmentation supervision |

## Realistic expectations (published SSS baselines)

Published YOLO-class results on SSS targets cluster around **0.74–0.92
mAP@0.5** depending on class difficulty and dataset — and degrade sharply on
cross-survey generalization. Rocks-vs-pipes remains the classic error mode;
that is exactly why the product positions multi-stage verification, not a
single detector, as the differentiator. **Never claim better numbers than
your own held-out evaluation.**

## Ghost nets: honest scoping

No public SSS dataset contains labelled ghost nets. Our approach:

1. Segment net-like fabric from FLS/other imagery where it exists.
2. Train net-like segmentation on **cut-paste synthesis**: real seabed tiles
   + rendered net textures + physics-consistent acoustic shadows
   (`backend/scripts/synth.py` renders the synthetic imagery; a training
   augmentation pipeline extends it).
3. Always label synthetic data as such; claim detection of *net-like
   filamentous debris*, never "real ghost nets", without real-world
   validation data.

## Reproducible splits

`python backend/scripts/split_dataset.py <images_dir> --by-folder …`
splits by survey group so adjacent-ping leakage does not inflate metrics.
Single-survey datasets (e.g. the watertank FLS folder) fall back to a
chronological split and print a loud leakage warning — report such numbers
with that caveat.
