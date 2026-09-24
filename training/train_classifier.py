"""Train the 64×64 natural-vs-artificial crop classifier (Colab GPU required).

Output contract (matches backend/app/pipeline/classify_natural.py):
- input: 1×1×64×64 grayscale, /255
- output: 2 logits [natural, artificial]; app takes softmax index 1 / prob[1]
  as p_artificial.

Metrics: accuracy, balanced accuracy, ROC-AUC — printed verbatim; no
synthetic numbers anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

B = Path("/content/aqua_train/bundle")
CLS = B / "cls" / "split"
RUNS = Path("/content/aqua_train/runs")


def require_gpu() -> None:
    if not torch.cuda.is_available():
        print("\nERROR: No GPU detected. Colab: Runtime > Change runtime type > GPU.\n", file=sys.stderr)
        raise SystemExit(3)
    print(f"GPU OK: {torch.cuda.get_device_name(0)}")


class CropDS(Dataset):
    def __init__(self, root: Path, train: bool):
        self.items = []
        for label, idx in (("natural", 0), ("artificial", 1)):
            for p in sorted((root / label).glob("*.png")):
                self.items.append((p, idx))
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import cv2

        p, y = self.items[i]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if self.train:
            if np.random.random() < 0.5:
                img = img[:, ::-1]
            if np.random.random() < 0.2:
                img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(img)[None]), y


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.h = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(self.f(x).flatten(1))


def evaluate(model: nn.Module, dl: DataLoader, device: str) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            p = torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
            ps.extend(p.tolist())
            ys.extend(y.tolist())
    preds = [int(p > 0.5) for p in ps]
    return {
        "accuracy": round(float(np.mean(np.array(preds) == np.array(ys))), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(ys, preds)), 4),
        "roc_auc": round(float(roc_auc_score(ys, ps)), 4),
        "n": len(ys),
    }


def main() -> None:
    require_gpu()
    cfg = json.loads((CLS / "meta.json").read_text(encoding="utf-8"))
    RUNS.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    tr = DataLoader(CropDS(CLS / "train", True), batch_size=128, shuffle=True, num_workers=2)
    va = DataLoader(CropDS(CLS / "val", False), batch_size=256, num_workers=2)

    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pw = torch.tensor(cfg.get("pos_weight_for_bce", 1.0), device=device)

    def lossf(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = torch.stack([torch.ones_like(pw), pw])[y]
        return F.cross_entropy(logits, y, weight=w)

    best, hist = -1.0, []
    for epoch in range(60):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss)
        m = evaluate(model, va, device)
        hist.append({"epoch": epoch, "loss": round(tot / len(tr), 4), **m})
        print(f"epoch {epoch:03d} loss={hist[-1]['loss']:.4f} val={json.dumps(m)}")
        if m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            torch.save(model.state_dict(), RUNS / "natart_best.pt")
    (RUNS / "classifier_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"best val balanced_accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
