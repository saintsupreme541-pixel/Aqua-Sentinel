"""Out-of-distribution anomaly scoring.

Deep classifiers are overconfident on inputs they have never seen.  To
honestly flag *unknown* objects we train a tiny patch autoencoder on
background (seabed) patches sampled from the image itself, then measure how
badly each detection's patches reconstruct.  High reconstruction error ⇒
the detection does not resemble the local background ⇒ anomaly.

This is a demonstration-grade, self-contained OOD detector: it needs no
external weights and runs in well under a second per image.  In production
it would be replaced by an encoder trained on a large survey-specific
background corpus.
"""

from __future__ import annotations

import numpy as np

PATCH = 16


class PatchAutoencoder:
    """Two-layer MLP autoencoder over flattened patches, pure numpy."""

    def __init__(self, patch: int = PATCH, hidden: int = 64, seed: int = 0):
        self.patch = patch
        self.in_dim = patch * patch
        rng = np.random.default_rng(seed)
        self.w1 = rng.standard_normal((self.in_dim, hidden)) * np.sqrt(2.0 / self.in_dim)
        self.b1 = np.zeros(hidden)
        self.w2 = rng.standard_normal((hidden, self.in_dim)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(self.in_dim)

    def _encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.w1 + self.b1)

    def _decode(self, h: np.ndarray) -> np.ndarray:
        return h @ self.w2 + self.b2

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._decode(self._encode(x))

    def fit(self, patches: np.ndarray, epochs: int = 80, lr: float = 0.02, batch: int = 128, seed: int = 0) -> float:
        x = np.asarray(patches, dtype=float)
        if x.ndim == 3:
            x = x.reshape(len(x), -1)
        n = len(x)
        if n == 0:
            return 0.0
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
        rng = np.random.default_rng(seed)
        last = 0.0
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                mb = x[idx[s : s + batch]]
                h = self._encode(mb)
                out = self._decode(h)
                err = out - mb
                dout = err / len(mb)
                dh = dout @ self.w2.T * (1 - h**2)
                self.w2 -= lr * h.T @ dout
                self.b2 -= lr * dout.sum(axis=0)
                self.w1 -= lr * mb.T @ dh
                self.b1 -= lr * dh.sum(axis=0)
            last = float(np.mean((self.forward(x) - x) ** 2))
        return last

    def errors(self, patches: np.ndarray) -> np.ndarray:
        x = np.asarray(patches, dtype=float)
        if x.ndim == 3:
            x = x.reshape(len(x), -1)
        if len(x) == 0:
            return np.array([])
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
        return np.mean((self.forward(x) - x) ** 2, axis=1)


def _patches_from(
    gray: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    count: int = 400,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    hgt, wid = gray.shape
    x0, y0, x1, y1 = region or (0, 0, wid, hgt)
    if x1 - x0 < PATCH or y1 - y0 < PATCH:
        return np.zeros((0, PATCH, PATCH), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(0)
    xs = rng.integers(x0, max(x0 + 1, x1 - PATCH + 1), size=count * 2)
    ys = rng.integers(y0, max(y0 + 1, y1 - PATCH + 1), size=count * 2)
    out = []
    for x, y in zip(xs, ys, strict=True):
        patch = gray[y : y + PATCH, x : x + PATCH]
        if patch.std() > 2.0:  # skip flat water-column patches
            out.append(patch.astype(np.float32))
        if len(out) >= count:
            break
    return np.asarray(out, dtype=np.float32)


def _box_to_region(box: dict, hgt: int, wid: int, margin: int = 8) -> tuple[int, int, int, int]:
    x0 = max(0, int(box["x"]) - margin)
    y0 = max(0, int(box["y"]) - margin)
    x1 = min(wid, int(box["x"] + box["w"]) + margin)
    y1 = min(hgt, int(box["y"] + box["h"]) + margin)
    return x0, y0, x1, y1


def anomaly_scores(gray: np.ndarray, boxes: list[dict], *, seed: int = 0) -> tuple[dict[str, float], dict]:
    """Per-box anomaly scores in [0, 1] (1 = strongly anomalous) plus meta.

    ``boxes``: list of dicts with x/y/w/h (``id`` optional).  Trains the
    autoencoder on background patches that avoid all detections.
    """
    hgt, wid = gray.shape
    rng = np.random.default_rng(seed)

    # background mask: everything outside the detection boxes (with margin)
    bg_mask = np.ones((hgt, wid), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = _box_to_region(b, hgt, wid)
        bg_mask[y0:y1, x0:x1] = False

    valid_ys, valid_xs = np.where(bg_mask)
    bg_patches: list[np.ndarray] = []
    attempts = 0
    while len(bg_patches) < 500 and attempts < 3000 and len(valid_ys) > 0:
        i = int(rng.integers(0, len(valid_ys)))
        y, x = int(valid_ys[i]), int(valid_xs[i])
        attempts += 1
        if x + PATCH > wid or y + PATCH > hgt:
            continue
        patch = gray[y : y + PATCH, x : x + PATCH]
        if patch.std() > 2.0:
            bg_patches.append(patch.astype(np.float32))

    ae = PatchAutoencoder(patch=PATCH)
    if len(bg_patches) < 40:
        meta = {"trained": False, "note": "too few background patches to build anomaly model"}
        return {b.get("id", str(i)): 0.5 for i, b in enumerate(boxes)}, meta
    ae.fit(np.asarray(bg_patches))
    bg_err = float(np.median(ae.errors(np.asarray(bg_patches)))) + 1e-6

    scores: dict[str, float] = {}
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = _box_to_region(b, hgt, wid)
        obj_patches = _patches_from(gray, (x0, y0, x1, y1), count=200, rng=rng)
        if len(obj_patches) == 0:
            scores[b.get("id", str(i))] = 0.5
            continue
        err = float(np.median(ae.errors(obj_patches)))
        score = min(1.0, max(0.0, (err - bg_err) / (6.0 * bg_err)))
        scores[b.get("id", str(i))] = round(score, 4)
    return scores, {"trained": True, "background_patches": len(bg_patches)}
