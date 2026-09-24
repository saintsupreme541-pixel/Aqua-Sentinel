"""Sonar-specific preprocessing.

Pipeline (each step records whether it ran, so every derived artifact is
reproducible from the raw input):

1. load & normalize      — 8-bit stretch to [0, 255]
2. optional despeckle    — median blur (standard) / fast NL-means (aggressive)
3. optional CLAHE        — contrast enhancement (off in the light preset)
4. optional TVG          — range-dependent gain compensation
5. slant-range correction— resample rows to uniform ground range
   (only when altitude + range metadata are present)

The default *light* preset deliberately avoids aggressive denoising: recent
studies show over-processing can destroy small targets in sonar imagery.
The raw input is always preserved separately (storage/raw).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import geolocate

# NOTE on slant-range correction: the geometry stages (shadow height,
# geolocation, dimensions) apply the analytic slant-to-ground conversion
# themselves, so row-warping the image here would double-correct and
# misalign pixel coordinates.  The warp remains available via
# `slant_range_correct()` for display-only use; presets do not apply it.
PRESETS: dict[str, dict[str, Any]] = {
    "light": {"despeckle": None, "clahe": False, "tvg": False},
    "standard": {"despeckle": "median3", "clahe": 2.0, "tvg": True},
    "aggressive": {"despeckle": "nlm", "clahe": 3.0, "tvg": True},
}


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load any image as 8-bit grayscale (handles 8/16-bit, RGB, RGBA)."""
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)  # unicode-safe on Windows
    img: Any = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode image: {p.name}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        if hi <= lo:
            return np.zeros_like(img).astype(np.uint8)
        img = ((img.astype(np.float32) - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    return img


def _normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img)
    out = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _tvg(gray: np.ndarray, range_m: float, altitude_m: float, alpha: float = 1.0) -> np.ndarray:
    """Compensate spherical spreading + absorption along the range axis.

    Applies a mild range-dependent gain; disabled when geometry is invalid.
    """
    hgt = gray.shape[0]
    slant = np.linspace(0.0, max(float(range_m), 1.0), hgt)
    ground = geolocate.slant_to_ground(slant, float(altitude_m))
    gain = np.clip((ground / np.maximum(ground.max(), 1e-6)) ** (2.0 * alpha), 0.5, 4.0)
    out = gray.astype(np.float32) * gain[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _slant_range_correct(gray: np.ndarray, range_m: float, altitude_m: float) -> tuple[np.ndarray, bool]:
    """Resample rows so that equal row spacing = equal ground range."""
    hgt, wid = gray.shape
    if altitude_m <= 0 or altitude_m >= range_m:
        return gray, False
    rows = np.arange(hgt, dtype=float)
    slant = range_m * rows / max(hgt - 1, 1)
    ground = geolocate.slant_to_ground(slant, altitude_m)
    if ground[-1] <= ground[0]:
        return gray, False
    grid = np.linspace(ground[0], ground[-1], hgt)
    src = np.interp(grid, ground, rows)
    idx = np.round(src).astype(int)
    idx = np.clip(idx, 0, hgt - 1)
    return gray[idx, :], True


def preprocess(gray: np.ndarray, meta: dict, preset: str = "light") -> tuple[np.ndarray, dict[str, Any]]:
    cfg = PRESETS.get(preset, PRESETS["light"])
    params: dict[str, Any] = {"preset": preset, "steps": [], "input_shape": list(gray.shape)}

    out = _normalize(gray)
    params["steps"].append("normalize(1-99 percentile stretch)")

    despeckle = cfg["despeckle"]
    if despeckle == "median3":
        out = cv2.medianBlur(out, 3)
        params["steps"].append("despeckle:median3")
    elif despeckle == "nlm":
        out = cv2.fastNlMeansDenoising(out, None, h=10, templateWindowSize=7, searchWindowSize=21)
        params["steps"].append("despeckle:fastNlMeans")

    if cfg["clahe"]:
        clahe = cv2.createCLAHE(clipLimit=cfg["clahe"], tileGridSize=(8, 8))
        out = clahe.apply(out)
        params["steps"].append(f"clahe(clip={cfg['clahe']})")

    if cfg["tvg"] and meta.get("range_m") and meta.get("altitude_m"):
        out = _tvg(out, float(meta["range_m"]), float(meta["altitude_m"]))
        params["steps"].append("tvg(range gain compensation)")

    if meta.get("slant_range_correct_display"):
        out, applied = _slant_range_correct(out, float(meta["range_m"]), float(meta["altitude_m"]))
        if applied:
            params["steps"].append("slant_range_correct(display)")
            params["slant_range_corrected"] = True

    params["output_shape"] = list(out.shape)
    return out, params


def save_preview(processed: np.ndarray, path: str | Path, detections: list[dict] | None = None) -> None:
    """Save the processed image (with optional detection boxes) as PNG."""
    vis = processed.copy()
    if detections:
        for d in detections:
            b = d["box"]
            cv2.rectangle(vis, (int(b["x"]), int(b["y"])), (int(b["x"] + b["w"]), int(b["y"] + b["h"])), (0, 255, 0), 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)
