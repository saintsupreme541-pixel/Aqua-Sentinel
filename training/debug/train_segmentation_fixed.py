"""Corrected U-Net training for binary wreck segmentation (repair run).

Reuses the EXACT ``UNet`` architecture from ``train_segmentation.py``. Every
delta vs the original trainer is listed here and motivated by the failure
investigation (deployed unet-sss.onnx predicts all-background, max sigmoid
~0.34 on in-domain wrecks):

D1  Foreground-aware tile sampling: each batch mixes ~50% tiles with
    >= fg_threshold foreground and ~50% uniform tiles. Original uniform
    sampling on ~1-2% foreground frames meant most optimization signal was
    "predict nothing".
D2  Loss rebalanced: 0.5*BCE + 0.5*(1 - soft_dice(eps=1e-5)). Original used
    1.0*BCE + (1 - soft_dice(eps=1.0)); with sparse targets the eps=1.0 dice
    term was ~0 loss gradient, so plain BCE drove all-background collapse.
D3  Positive-bias initialization of the final conv layer (bias = log-prior of
    foreground, small) so training starts near the true class balance instead
    of a strong all-background prior.
D4  Gradient clipping (max-norm 5.0) + finite-loss guard (original reached
    loss=nan on Colab).
D5  No AMP (fp32, deterministic). Original used autocast (reported unstable).
D6  Same site-level split (seed 20260905) and augmentation policy as the
    original; images /255; masks binary {0,1} float targets.
D7  Best checkpoint by validation Dice computed with the tiled sliding-window
    protocol at 0.5 (same as original validate()); history JSON saved.
D8  Multi-view sampling: with probability ``--fullframe-prob`` (default 0.25)
    a training sample is the WHOLE image resized to 512x512 (mask nearest-
    resized) instead of a native-scale tile. The deployed app adapter (fixed
    contract, unchangeable) feeds exactly this full-frame-resized view, so the
    model must respond at that scale too; tiles keep native-scale detail.

Usage (CPU or CUDA, auto-detected):
    python train_segmentation_fixed.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys_path = HERE.parent
import sys  # noqa: E402

sys.path.insert(0, str(sys_path))

import train_segmentation as _ts  # noqa: E402
from train_segmentation import UNet, _read_gray, _read_mask, validate  # noqa: E402

SEG = HERE.parent.parent / "data" / "work" / "unet_investigation" / "bundle_seg"
OUTDIR = HERE.parent.parent / "data" / "work" / "unet_investigation" / "repair"
_ts.SEG = SEG  # the original module hardcodes /content/aqua_train/bundle (Colab) — rebind to the local extraction
SEED = 20260905
TILE = 512


def site_split(val_frac: float = 0.15, seed: int = SEED) -> tuple[list[str], list[str]]:
    """EXACT reproduction of prepare_segmentation.py::site_split (stdlib RNG).

    The original used ``random.Random(seed).shuffle`` — a numpy permutation
    draws different val sites, which would silently change the benchmark.
    """
    smap = json.loads((SEG / "site_map.json").read_text(encoding="utf-8"))
    sites = sorted(set(smap.values()))
    rng = random.Random(seed)
    rng.shuffle(sites)
    n_val = max(1, round(len(sites) * val_frac))
    val_sites = set(sites[:n_val])
    tr, va = [], []
    for stem, site in sorted(smap.items()):
        (va if site in val_sites else tr).append(stem)
    return tr, va


class TileIndex:
    """Precomputed tile table: (stem, y, x, fg_frac)."""

    def __init__(self, stems: list[str], tile: int, min_fg: float):
        rng = np.random.default_rng(SEED + 1)
        self.entries: list[tuple[str, int, int, float]] = []
        for s in stems:
            msk = _read_mask(SEG / "masks" / f"{s}.png")
            h, w = msk.shape
            for _ in range(200):
                y = int(rng.integers(0, max(h - tile, 1)))
                x = int(rng.integers(0, max(w - tile, 1)))
                frac = float((msk[y : y + tile, x : x + tile] > 0).mean())
                self.entries.append((s, y, x, frac))

    def fg_pool(self, min_fg: float) -> list[tuple[str, int, int, float]]:
        return [e for e in self.entries if e[3] >= min_fg]

    def all_pool(self) -> list[tuple[str, int, int, float]]:
        return list(self.entries)


def load_tile(s: str, y: int, x: int, tile: int, rng: np.random.Generator | None, train: bool):
    img = _read_gray(SEG / "images" / f"{s}.png")
    msk = _read_mask(SEG / "masks" / f"{s}.png")
    img_t = img[y : y + tile, x : x + tile]
    msk_t = msk[y : y + tile, x : x + tile]
    pad_h, pad_w = tile - img_t.shape[0], tile - img_t.shape[1]
    if pad_h or pad_w:
        img_t = np.pad(img_t, ((0, pad_h), (0, pad_w)))
        msk_t = np.pad(msk_t, ((0, pad_h), (0, pad_w)))
    if train and rng is not None:
        if rng.random() < 0.5:
            img_t, msk_t = img_t[:, ::-1], msk_t[:, ::-1]
        if rng.random() < 0.15:
            img_t, msk_t = img_t[::-1, :], msk_t[::-1, :]
        if rng.random() < 0.5:
            k = int(rng.integers(0, 4))
            img_t, msk_t = np.rot90(img_t, k), np.rot90(msk_t, k)
        if rng.random() < 0.2:
            img_t = np.clip(img_t * (1 + rng.uniform(-0.2, 0.2)), 0, 255)
    img_t = np.ascontiguousarray(img_t, dtype=np.float32) / 255.0
    msk_t = np.ascontiguousarray(msk_t, dtype=np.float32)
    return img_t[None], msk_t[None]


def load_sample(s: str, y: int, x: int, tile: int, rng: np.random.Generator | None, train: bool):
    """Full-frame-resized view (D8) when s == '__fullframe__', else a tile.

    In the fullframe branch the stem travels in ``y`` so every batch entry
    keeps the same 4-tuple shape.
    """
    if s != "__fullframe__":
        return load_tile(s, y, x, tile, rng, train)
    import cv2

    img = _read_gray(SEG / "images" / f"{y}.png")
    msk = _read_mask(SEG / "masks" / f"{y}.png")
    img_r = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
    msk_r = cv2.resize(msk, (tile, tile), interpolation=cv2.INTER_NEAREST)
    if train and rng is not None:
        if rng.random() < 0.5:
            img_r, msk_r = img_r[:, ::-1], msk_r[:, ::-1]
        if rng.random() < 0.5:
            k = int(rng.integers(0, 4))
            img_r, msk_r = np.rot90(img_r, k), np.rot90(msk_r, k)
    img_r = np.ascontiguousarray(img_r, dtype=np.float32) / 255.0
    msk_r = np.ascontiguousarray(msk_r, dtype=np.float32)
    return img_r[None], msk_r[None]


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    return ((2 * inter + eps) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-fg", type=float, default=0.01)
    ap.add_argument("--fullframe-prob", type=float, default=0.25)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-every", type=int, default=2, help="validate every N epochs (val is slow)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = args.device
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})", flush=True)

    tr, va = site_split()
    print(f"site split: train={len(tr)} val={len(va)}", flush=True)
    (OUTDIR / "reproduced_split.json").write_text(
        json.dumps({"train": tr, "val": va}, indent=2), encoding="utf-8"
    )

    idx = TileIndex(tr, TILE, args.min_fg)
    fg = idx.fg_pool(args.min_fg)
    allE = idx.all_pool()
    print(f"tile table: {len(allE)} uniform, {len(fg)} fg-bearing (>= {args.min_fg:.0%})", flush=True)

    model = UNet().to(device)
    # D3: bias the final 1x1 conv toward the log-prior of foreground presence
    with torch.no_grad():
        prior = 0.02  # conservative start; tiles are ~50% fg-bearing in batch
        model.out.bias.fill_(math.log(prior / (1 - prior)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED + 2)

    best = -1.0
    best_epoch = -1
    hist: list[dict] = []
    t0 = time.time()
    step_in_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        n_nan = 0
        for _ in range(args.steps_per_epoch):
            batch = []
            for _b in range(args.batch):
                if rng.random() < args.fullframe_prob:
                    batch.append(("__fullframe__", tr[int(rng.integers(0, len(tr)))], 0, 0.0))
                elif rng.random() < 0.5 and fg:
                    batch.append(fg[int(rng.integers(0, len(fg)))])
                else:
                    batch.append(allE[int(rng.integers(0, len(allE)))])
            imgs, msks = zip(*(load_sample(s, y, x, TILE, rng, True) for s, y, x, _f in batch))
            xi = torch.from_numpy(np.stack(imgs)).to(device)
            yi = torch.from_numpy(np.stack(msks)).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xi)
            loss = 0.5 * bce(logits, yi) + 0.5 * (1 - soft_dice(logits, yi))
            if not torch.isfinite(loss):
                n_nan += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            step_in_epoch += 1
        train_loss = tot / max(args.steps_per_epoch - n_nan, 1)

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            m = validate(model, va, TILE, device)
            hist.append({"epoch": epoch, "train_loss": round(train_loss, 4), **m})
            print(
                f"epoch {epoch:03d}  loss={train_loss:.4f}  val_dice={m['dice']:.4f}  val_iou={m['iou']:.4f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            if m["dice"] > best:
                best = m["dice"]
                best_epoch = epoch
                torch.save(
                    {"state_dict": model.state_dict(), "base": 32, "val_dice": best, "val_iou": m["iou"], "epoch": epoch},
                    OUTDIR / "unet_repair_best.pt",
                )
            if epoch - best_epoch > args.patience:
                print(f"early stop at epoch {epoch} (best dice {best:.4f} @ {best_epoch})", flush=True)
                break
        else:
            hist.append({"epoch": epoch, "train_loss": round(train_loss, 4)})
            print(f"epoch {epoch:03d}  loss={train_loss:.4f}  (val skipped)", flush=True)

    (OUTDIR / "repair_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"BEST val dice: {best:.4f} @ epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
