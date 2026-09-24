"""Synthetic Side-Scan Sonar image generator.

Used for (a) demo material, (b) offline tests, (c) the evaluation harness.
Images are labelled as SYNTHETIC wherever they surface — never presented as
real survey data.  The generator mimics the *appearance* of sonar: seabed
texture + speckle + range gain falloff + nadir band + objects with acoustic
shadows, so the pipeline's geometry modules can be exercised end-to-end.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

CLASSES = ("pipe", "ghost_net", "wreck", "rock", "cylinder", "debris")


def make_seabed(shape: tuple[int, int], seed: int, ripple: bool = True) -> np.ndarray:
    """Seabed texture + sand ripples + *realistic* spatially-correlated speckle.

    Speckle is log-normal with ~12% amplitude and ~4 px correlation — close to
    the appearance of despeckled survey imagery, not the shot noise some
    generators use.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    base = rng.normal(0, 1, (h // 16 + 2, w // 16 + 2)).astype(np.float32)
    smooth = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(smooth, (0, 0), 12)
    frac = (smooth - smooth.min()) / (np.ptp(smooth) + 1e-6)
    img = 48.0 + 52.0 * frac  # moderate albedo band 48–100

    if ripple:  # sand ripples along the across-track axis
        x = np.arange(w, dtype=np.float32)
        for _ in range(2):
            freq = rng.uniform(0.005, 0.013)
            amp = rng.uniform(4, 9)
            phase = rng.uniform(0, math.tau)
            img += amp * np.sin(2 * math.pi * freq * x + phase)[None, :]

    # correlated multiplicative speckle (~12% amplitude)
    noise = rng.standard_normal((h // 4 + 1, w // 4 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    speckle = np.exp(0.12 * noise).astype(np.float32)
    img = img * speckle

    # range gain falloff: darker with increasing row (range)
    rows = np.linspace(0.92, 1.10, h, dtype=np.float32)[:, None]
    img = img * rows

    # nadir gap: directly under the towfish there is no seafloor return
    nw = int(w * 0.05)
    x0 = w // 2 - nw // 2
    img[:, x0 : x0 + nw] *= 0.35

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _shadow(img: np.ndarray, x, y, w, h, seed: int, strength: float = 0.55) -> None:
    """Draw a dark acoustic shadow directly below a highlight (flat seabed)."""
    rng = np.random.default_rng(seed)
    H, W = img.shape
    sh = max(6, int(h * rng.uniform(0.7, 1.6)))
    sw = max(4, int(w * rng.uniform(0.7, 0.95)))
    sx = min(W - 1, max(0, x + (w - sw) // 2))
    sy = y + h
    sy1 = min(H, sy + sh)
    sx1 = min(W, sx + sw)
    panel = img[sy:sy1, sx:sx1].astype(np.float32)
    panel *= 1.0 - strength
    img[sy:sy1, sx:sx1] = panel.astype(np.uint8)


def add_pipe(img: np.ndarray, box: tuple[int, int, int, int], seed: int) -> None:
    x, y, w, h = box
    cv2.ellipse(img, ((x + w // 2, y + h // 2), (w, h), 0), 212, -1)
    # specular core: bright elongated return along the cylinder top
    core_w = max(3, w // 3)
    cv2.ellipse(img, ((x + w // 2, y + h // 2), (core_w, h), 0), 235, -1)
    _shadow(img, x, y, w, h, seed)


def add_ghost_net(img: np.ndarray, box: tuple[int, int, int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, y, w, h = box
    x2, y2 = x + w, y + h
    # filament network
    for _ in range(rng.integers(9, 15)):
        px, py = rng.integers(x, x2), rng.integers(y, y2)
        pts = [(px, py)]
        for _ in range(rng.integers(6, 11)):
            px += int(rng.normal(0, w / 8))
            py += int(rng.normal(0, h / 8))
            pts.append((int(np.clip(px, x, x2)), int(np.clip(py, y, y2))))
        cv2.polylines(img, [np.array(pts, np.int32)], False, int(rng.integers(205, 245)), 2, cv2.LINE_AA)
    # dense mesh patch in the centre
    for i in range(6):
        xx = x + int(w * (0.3 + 0.08 * i))
        for j in range(5):
            yy = y + int(h * (0.3 + 0.1 * j))
            cv2.circle(img, (xx, yy), 2, 190, -1)
    _shadow(img, x, y, w, h, seed, strength=0.35)  # nets cast faint, broken shadows


def add_wreck(img: np.ndarray, box: tuple[int, int, int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, y, w, h = box
    # hull: elongated bright shape with blocky structure
    cv2.rectangle(img, (x, y), (x + w, y + int(h * 0.45)), int(rng.integers(180, 215)), -1)
    for _ in range(rng.integers(3, 6)):
        bx = x + rng.integers(0, w - w // 5)
        by = y + int(h * 0.4) + rng.integers(0, int(h * 0.4))
        cv2.rectangle(
            img,
            (int(bx), int(by)),
            (int(bx + rng.integers(w // 5, w // 3)), int(by + rng.integers(4, 9))),
            int(rng.integers(160, 200)),
            -1,
        )
    _shadow(img, x, y, w, h, seed)


def add_rock(img: np.ndarray, box: tuple[int, int, int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, y, w, h = box
    pts = []
    cx, cy = x + w / 2, y + h / 2
    for k in range(10):
        ang = 2 * math.pi * k / 10
        rr = rng.uniform(0.55, 0.95)
        pts.append((int(cx + rr * (w / 2) * math.cos(ang)), int(cy + rr * (h / 2) * math.sin(ang))))
    pts = np.array(pts, np.int32)
    # mottled fill: irregular, mid brightness, NO strong highlight
    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    panel = img.copy()
    tones = rng.integers(95, 135, size=img.shape).astype(np.uint8)
    panel = np.where(mask > 0, tones, panel)
    # texture speckles on the rock
    rock_pts = np.argwhere(mask > 0)
    if len(rock_pts):
        idx = rng.integers(0, len(rock_pts), size=min(400, len(rock_pts)))
        for i in idx:
            py, px = rock_pts[i]
            panel[py, px] = 90 + rng.integers(0, 60)
    img[...] = np.where(mask > 0, panel, img)


def add_cylinder(img: np.ndarray, box: tuple[int, int, int, int], seed: int) -> None:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), 215, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), 232, 2)
    cv2.ellipse(img, ((x + w // 2, y + h // 2), (w, int(h * 0.6)), 0), 222, -1)
    _shadow(img, x, y, w, h, seed)


def add_debris(img: np.ndarray, box: tuple[int, int, int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), int(rng.integers(150, 200)), -1)
    _shadow(img, x, y, w, h, seed, strength=0.45)


OPERATIONS = {
    "pipe": add_pipe,
    "ghost_net": add_ghost_net,
    "wreck": add_wreck,
    "rock": add_rock,
    "cylinder": add_cylinder,
    "debris": add_debris,
}


def render(kind: str, seed: int, shape: tuple[int, int] = (480, 512)) -> tuple[np.ndarray, dict]:
    """Render one synthetic SSS scene. Returns (image, ground-truth dict)."""
    img = make_seabed(shape, seed)
    rng = np.random.default_rng(seed + 999)
    kde = kind
    gt_boxes = []
    if kde == "ghost_net":
        w, h = int(shape[1] * 0.18), int(shape[0] * 0.16)
        x, y = (
            rng.integers(30, shape[1] - w - 30),
            rng.integers(int(shape[0] * 0.45), shape[0] - h - int(shape[0] * 0.12)),
        )
    elif kde == "rock":
        w, h = int(shape[1] * 0.10), int(shape[0] * 0.08)
        x, y = (
            rng.integers(30, shape[1] - w - 30),
            rng.integers(int(shape[0] * 0.4), shape[0] - h - int(shape[0] * 0.1)),
        )
    else:
        w, h = int(shape[1] * 0.16), int(shape[0] * 0.22)
        x, y = (
            rng.integers(30, shape[1] - w - 30),
            rng.integers(int(shape[0] * 0.45), shape[0] - h - int(shape[0] * 0.12)),
        )
    OPERATIONS[kde](img, (int(x), int(y), int(w), int(h)), int(seed))
    gt_boxes.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h), "class": kde})
    gt = {"kind": kde, "seed": int(seed), "shape": list(shape), "boxes": gt_boxes}
    return img, gt


def synthetic_metadata(
    index: int, kind: str, lat0: float = 17.6800, lon0: float = 83.3100, heading: float = 90.0
) -> dict[str, Any]:
    """Plausible synthetic survey metadata (explicitly synthetic — never real GPS)."""
    along_m = 40.0 * index  # 40 m between pings along track (> consistency threshold)
    lat = lat0 + (along_m / 111320.0) * math.cos(math.radians(heading - 90))
    lon = lon0 + (along_m / (111320.0 * math.cos(math.radians(lat0)))) * math.sin(math.radians(heading - 90))
    return {
        "synthetic": True,
        "sonar_type": "sss",
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "heading_deg": heading,
        "altitude_m": 15.0,
        "range_m": 60.0,
        "side": "starboard",
        "preprocess_preset": "light",
    }
