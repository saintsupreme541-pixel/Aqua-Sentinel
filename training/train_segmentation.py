"""Train a U-Net for binary wreck segmentation (Colab GPU required).

- Full-image training would blow memory on 1728×18000+ waterfalls; we train
  on 512×512 random tiles from site-split train images and validate on
  whole-image sliding windows of val sites.
- Exports a pure-torch state_dict + a scripted .pt; ONNX export happens in
  export_onnx.py so this file stays training-only.
- Metrics: Dice + IoU computed at 0.5 threshold, printed verbatim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

B = Path("/content/aqua_train/bundle")
SEG = B / "seg" / "ai4shipwrecks"
RUNS = Path("/content/aqua_train/runs")


def require_gpu() -> None:
    if not torch.cuda.is_available():
        print(
            "\nERROR: No GPU detected. In Colab: Runtime > Change runtime type > GPU, then re-run.\n"
            "U-Net training on CPU is deliberately not supported by this script.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    print(f"GPU OK: {torch.cuda.get_device_name(0)}")


class UNet(nn.Module):
    """Minimal U-Net, 1-channel in / 1-channel logits out (app-compatible)."""

    def __init__(self, base: int = 32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.d1 = blk(1, base)
        self.d2 = blk(base, base * 2)
        self.d3 = blk(base * 2, base * 4)
        self.d4 = blk(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = blk(base * 8 + base * 4, base * 4)
        self.u2 = blk(base * 4 + base * 2, base * 2)
        self.u1 = blk(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        u3 = F.interpolate(d4, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.u3(torch.cat([u3, d3], 1))
        u2 = F.interpolate(u3, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, d2], 1))
        u1 = F.interpolate(u2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, d1], 1))
        return self.out(u1)


class TileDataset(Dataset):
    def __init__(self, stems: list[str], tile: int, train: bool):
        self.items = [(SEG / "images" / f"{s}.png", SEG / "masks" / f"{s}.png") for s in stems]
        self.tile = tile
        self.train = train

    def __len__(self) -> int:
        return len(self.items) * (24 if self.train else 4)

    def __getitem__(self, i: int):
        img_p, msk_p = self.items[i % len(self.items)]
        img = _read_gray(img_p)
        msk = _read_mask(msk_p)
        h, w = img.shape
        t = self.tile
        rng = np.random.default_rng(i)
        y = int(rng.integers(0, max(h - t, 1)))
        x = int(rng.integers(0, max(w - t, 1)))
        img_t = img[y : y + t, x : x + t]
        msk_t = msk[y : y + t, x : x + t]
        # pad partial tiles
        pad_h, pad_w = t - img_t.shape[0], t - img_t.shape[1]
        if pad_h or pad_w:
            img_t = np.pad(img_t, ((0, pad_h), (0, pad_w)))
            msk_t = np.pad(msk_t, ((0, pad_h), (0, pad_w)))
        if self.train:
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
        return torch.from_numpy(img_t[None]), torch.from_numpy(msk_t[None])


def _read_gray(p: Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return im


def _read_mask(p: Path) -> np.ndarray:
    import cv2

    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.uint8)


@torch.no_grad()
def validate(model: nn.Module, stems: list[str], tile: int, device: str) -> dict:
    model.eval()
    inter = union = pred_pos = true_pos = 0
    for s in stems:
        img = _read_gray(SEG / "images" / f"{s}.png")
        msk = _read_mask(SEG / "masks" / f"{s}.png")
        h, w = img.shape
        # stride tiles with 25% overlap
        probs = np.zeros((h, w), np.float32)
        count = np.zeros((h, w), np.float32)
        step = int(tile * 0.75)
        for y in range(0, max(h - tile, 0) + 1, step):
            for x in range(0, max(w - tile, 0) + 1, step):
                yt, xt = min(y, h - tile), min(x, w - tile)
                t = torch.from_numpy(img[yt : yt + tile, xt : xt + tile][None, None].astype(np.float32) / 255.0).to(
                    device
                )
                with torch.autocast(device_type=device[:3] if device.startswith("cuda") else "cpu"):
                    p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
                probs[yt : yt + tile, xt : xt + tile] += p
                count[yt : yt + tile, xt : xt + tile] += 1
        probs = probs / np.maximum(count, 1)
        pred = probs > 0.5
        gt = msk > 0
        inter += int((pred & gt).sum())
        union += int((pred | gt).sum())
        pred_pos += int(pred.sum())
        true_pos += int(gt.sum())
    dice = 2 * inter / max(pred_pos + true_pos, 1)
    iou = inter / max(union, 1)
    return {"dice": round(dice, 4), "iou": round(iou, 4)}


def main() -> None:
    require_gpu()
    cfg = yaml.safe_load((B / "config.yaml").read_text(encoding="utf-8"))
    scfg = cfg["segmentation"]
    split = json.loads((SEG / "split.json").read_text(encoding="utf-8"))
    RUNS.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    tile = scfg["imgsz"]

    tr_ds = TileDataset(split["train"], tile, train=True)
    va_stems = split["val"]
    tr_dl = DataLoader(tr_ds, batch_size=scfg["batch"], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=scfg["lr"])
    # positives are rare -> dice+bce blend handles imbalance without weights
    bce = nn.BCEWithLogitsLoss()

    best = -1.0
    best_epoch = -1
    hist = []
    for epoch in range(scfg["epochs"]):
        model.train()
        tot = 0.0
        for img, msk in tr_dl:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(img)
            loss = bce(logits, msk) + (1 - _soft_dice(logits, msk))
            loss.backward()
            opt.step()
            tot += float(loss)
        m = validate(model, va_stems, tile, device)
        hist.append({"epoch": epoch, "loss": round(tot / max(len(tr_dl), 1), 4), **m})
        print(f"epoch {epoch:03d}  loss={hist[-1]['loss']:.4f}  val_dice={m['dice']:.4f}  val_iou={m['iou']:.4f}")
        if m["dice"] > best:
            best = m["dice"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "base": 32, "val_dice": best}, RUNS / "unet_best.pt")
        if epoch - best_epoch > scfg["patience"]:
            print(f"early stop at epoch {epoch} (best dice {best:.4f} @ epoch {best_epoch})")
            break
    (RUNS / "segmentation_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"best val dice: {best:.4f}")


def _soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    return ((2 * inter + eps) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)).mean()


if __name__ == "__main__":
    main()
