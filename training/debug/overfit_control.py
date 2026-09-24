"""Overfit control test for the wreck-segmentation U-Net (CPU-friendly).

Question: can the *training implementation* memorize a handful of samples?
If a fresh U-Net cannot reach near-perfect Dice on the very tiles it trains
on, the pipeline (loss / targets / optimization) is broken and full training
is pointless.

This script deliberately reuses the EXACT ``UNet`` and mask-reading code from
``training/train_segmentation.py`` (imported, not copied) so the result speaks
about the production pipeline. Differences vs the Colab trainer, all deliberate
and documented:

1. fixed tile set: every tile contains >= min_fg foreground (plus a few
   background tiles) instead of uniformly random tiles — memorization is the
   objective, not generalization;
2. loss = BCEWithLogitsLoss + (1 - soft_dice) with eps=1e-5 (the Colab trainer
   used eps=1.0, which lets the Dice term vanish on sparse targets);
3. gradient clipping (max-norm 5.0) + finite-loss guard (the Colab run
   reportedly reached loss=nan);
4. no AMP (deterministic fp32 on CPU);
5. seed fixed; evaluates Dice/IoU on the SAME fixed tile set at 0.5.

Usage:
    python overfit_control.py --steps 400 --batch 4 --tile 512
    python overfit_control.py --selftest   # tiny 20-step smoke of the harness
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # training/

from train_segmentation import UNet, _read_gray, _read_mask  # noqa: E402  (production classes)

SEG = HERE.parent.parent / "data" / "work" / "unet_investigation" / "bundle_seg"
OUT = HERE.parent.parent / "data" / "work" / "unet_investigation" / "overfit"
SEED = 20260905


def pick_stems(k: int) -> list[str]:
    import cv2

    smap = json.loads((SEG / "site_map.json").read_text(encoding="utf-8"))
    scored = []
    for s in sorted(smap):
        m = _read_mask(SEG / "masks" / f"{s}.png")
        scored.append((float((m > 0).mean()), s))
    scored.sort(reverse=True)
    # spread across distinct sites, prefer healthy foreground fractions
    chosen, seen = [], set()
    for frac, s in scored:
        site = smap[s]
        if site in seen:
            continue
        seen.add(site)
        chosen.append(s)
        if len(chosen) == k:
            break
    return chosen


def build_fixed_tiles(stems: list[str], tile: int, per_img: int, min_fg: float, bg_per_img: int):
    """Deterministic tile list: foreground-bearing tiles + a few background."""
    import cv2

    rng = np.random.default_rng(SEED)
    tiles = []
    for s in stems:
        img = _read_gray(SEG / "images" / f"{s}.png")
        msk = _read_mask(SEG / "masks" / f"{s}.png")
        h, w = img.shape
        fg_tiles, bg_tiles = [], []
        for _ in range(per_img * 40):
            y = int(rng.integers(0, max(h - tile, 1)))
            x = int(rng.integers(0, max(w - tile, 1)))
            frac = float((msk[y : y + tile, x : x + tile] > 0).mean())
            (fg_tiles if frac >= min_fg else bg_tiles).append((y, x, frac))
            if len(fg_tiles) >= per_img and len(bg_tiles) >= bg_per_img:
                break
        fg_tiles.sort(key=lambda t: -t[2])
        for y, x, _ in fg_tiles[:per_img]:
            tiles.append((s, y, x))
        for y, x, _ in bg_tiles[:bg_per_img]:
            tiles.append((s, y, x))
    return tiles


def load_tile(s: str, y: int, x: int, tile: int, augment_rng: np.random.Generator | None):
    img = _read_gray(SEG / "images" / f"{s}.png")
    msk = _read_mask(SEG / "masks" / f"{s}.png")
    h, w = img.shape
    img_t = img[y : y + tile, x : x + tile]
    msk_t = msk[y : y + tile, x : x + tile]
    pad_h, pad_w = tile - img_t.shape[0], tile - img_t.shape[1]
    if pad_h or pad_w:
        img_t = np.pad(img_t, ((0, pad_h), (0, pad_w)))
        msk_t = np.pad(msk_t, ((0, pad_h), (0, pad_w)))
    if augment_rng is not None:
        if augment_rng.random() < 0.5:
            img_t, msk_t = img_t[:, ::-1], msk_t[:, ::-1]
        if augment_rng.random() < 0.5:
            k = int(augment_rng.integers(0, 4))
            img_t, msk_t = np.rot90(img_t, k), np.rot90(msk_t, k)
    img_t = np.ascontiguousarray(img_t, dtype=np.float32) / 255.0
    msk_t = np.ascontiguousarray(msk_t, dtype=np.float32)
    return img_t[None], msk_t[None]


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    return ((2 * inter + eps) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)).mean()


def evaluate(model: UNet, tiles, tile: int, device: str) -> dict:
    model.eval()
    inter = union = pp = tp = 0
    with torch.no_grad():
        for s, y, x in tiles:
            img_t, msk_t = load_tile(s, y, x, tile, None)
            xi = torch.from_numpy(img_t[None]).to(device)
            lg = model(xi)[0, 0]
            pr = (torch.sigmoid(lg) > 0.5).cpu().numpy()
            gt = msk_t[0] > 0.5
            inter += int((pr & gt).sum())
            union += int((pr | gt).sum())
            pp += int(pr.sum())
            tp += int(gt.sum())
    model.train()
    return {
        "dice": 2 * inter / max(pp + tp, 1),
        "iou": inter / max(union, 1),
        "precision": inter / max(pp, 1),
        "recall": inter / max(tp, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per-img", type=int, default=24)
    ap.add_argument("--bg-per-img", type=int, default=4)
    ap.add_argument("--min-fg", type=float, default=0.01)
    ap.add_argument("--n-imgs", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--selftest", action="store_true", help="20-step smoke of the harness")
    ap.add_argument("--save", default=str(OUT / "overfit_unet.pt"))
    args = ap.parse_args()
    if args.selftest:
        args.steps, args.per_img, args.bg_per_img = 20, 4, 1

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}", flush=True)

    stems = pick_stems(args.n_imgs)
    print(f"stems ({len(stems)}):", stems)
    tiles = build_fixed_tiles(stems, args.tile, args.per_img, args.min_fg, args.bg_per_img)
    fg_tiles = sum(1 for s, y, x in tiles if (_read_mask(SEG / 'masks' / f'{s}.png')[y:y+args.tile, x:x+args.tile] > 0).mean() >= args.min_fg)
    print(f"tiles: {len(tiles)} ({fg_tiles} foreground-bearing >= {args.min_fg:.0%})")

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    step = 0
    losses: list[float] = []
    while step < args.steps:
        idx = rng.permutation(len(tiles))
        for i in range(0, len(idx) - args.batch + 1, args.batch):
            if step >= args.steps:
                break
            batch = [tiles[j] for j in idx[i : i + args.batch]]
            imgs, msks = zip(*(load_tile(s, y, x, args.tile, rng) for s, y, x in batch))
            xi = torch.from_numpy(np.stack(imgs)).to(device)
            yi = torch.from_numpy(np.stack(msks)).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xi)
            loss = bce(logits, yi) + (1 - soft_dice(logits, yi))
            if not torch.isfinite(loss):
                print(f"NON-FINITE LOSS at step {step} — aborting")
                sys.exit(2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"step {step:4d}  loss={losses[-1]:.4f}  ({step / (time.time() - t0):.2f} it/s)")

    m = evaluate(model, tiles, args.tile, device)
    frac_pos = float(np.mean([(_read_mask(SEG / 'masks' / f'{s}.png')[y:y+args.tile, x:x+args.tile] > 0).mean() for s, y, x in tiles]))
    print(f"\n=== OVERFIT RESULT ({args.steps} steps, tile {args.tile}) ===")
    print(f"train loss: first={losses[0]:.4f} last={losses[-1]:.4f} min={min(losses):.4f}")
    print(f"same-tile dice={m['dice']:.4f} iou={m['iou']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")
    print(f"mean GT foreground fraction of eval tiles: {frac_pos:.4f}")
    verdict = "PASS — pipeline can memorize" if m["dice"] >= 0.8 else "FAIL — pipeline cannot memorize (investigate before full training)"
    print(verdict)

    torch.save({"state_dict": model.state_dict(), "base": 32, "tiles": args.tile, "stems": stems}, args.save)
    (OUT / "overfit_result.json").write_text(
        json.dumps({"steps": args.steps, "tile": args.tile, "stems": stems, "loss_first": losses[0],
                    "loss_last": losses[-1], **m, "gt_fg_frac": frac_pos, "verdict": verdict}, indent=2),
        encoding="utf-8",
    )
    print("saved:", args.save)


if __name__ == "__main__":
    main()
