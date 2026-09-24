"""Verify GT mask alignment on AI4Shipwrecks seg samples.

For each sample: image/mask dimension equality, positive fraction, mask
bounding region, and a saved overlay PNG (raw image + GT mask tinted green +
wreck-region bounding box). Overlays land in
data/work/unet_investigation/overlays/gt/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SEG = HERE.parent.parent / "data" / "work" / "unet_investigation" / "bundle_seg"
OUT = HERE.parent.parent / "data" / "work" / "unet_investigation" / "overlays" / "gt"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = sys.argv[1:] or ["Montana_07", "EB_Allen_07", "Corsair_02", "Pewabic_03", "DM_Wilson_15"]
    ok = True
    for stem in samples:
        ip, mp = SEG / "images" / f"{stem}.png", SEG / "masks" / f"{stem}.png"
        if not ip.exists() or not mp.exists():
            print(f"{stem}: MISSING FILES")
            ok = False
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        b = msk > 0
        frac = float(b.mean())
        ys, xs = np.where(b)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if b.any() else None
        # alignment sanity: wreck pixels should be brighter than image median
        med = float(np.median(img))
        bright_in = float(np.median(img[b])) if b.any() else float("nan")
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[b] = (0.4 * vis[b] + np.array([0, 180, 0]) * 0.6).astype(np.uint8)
        if bbox:
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{stem}_gt_overlay.png"), vis)
        aligned = b.any() and bright_in > med
        ok &= bool(aligned)
        print(
            f"{stem}: img={img.shape} mask={msk.shape} dims_match={img.shape == msk.shape} "
            f"pos_frac={frac:.4f} bbox={bbox} mask_vals={sorted(np.unique(msk).tolist())} "
            f"median_gray={med:.1f} median_in_mask={bright_in:.1f} aligned={aligned}"
        )
    print("OVERALL:", "ALIGNED" if ok else "CHECK FAILURES ABOVE")


if __name__ == "__main__":
    main()
