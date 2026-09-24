"""Pluggable semantic segmentation.

Backends: ``onnx`` (U-Net/DeepLab export, 1×C×H×W logits or single-class
mask), ``ultralytics`` (YOLO-seg), ``heuristic`` (threshold fallback).
Every backend returns a full-size uint8 binary mask + the mask/box area
fraction used by the segmentation-agreement evidence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..fallback.segment_threshold import segment_heuristic
from ..models.registry import load_registry, resolve

log = logging.getLogger("aqua.segment")


class ONNXSegmenter:
    def __init__(self, cfg: dict):
        import onnxruntime as ort

        self.session = ort.InferenceSession(cfg["path"], providers=["CPUExecutionProvider"])
        self.nc = int(cfg.get("num_classes", 1))

    def segment(self, gray: np.ndarray, box: dict) -> tuple[np.ndarray, float]:
        hgt, wid = gray.shape
        # The exported U-Net has a STATIC input (1, 1, 512, 512): resize the
        # full-resolution image in, run, then resize the logits back to the
        # original size before thresholding — never crop, never fail on size.
        import cv2

        inp = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_LINEAR)
        out = self.session.run(
            None, {self.session.get_inputs()[0].name: inp[None, None, :, :].astype(np.float32) / 255.0}
        )[0]
        out = np.asarray(out)
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 3 and out.shape[0] == 1:
            mask = out[0]
        elif out.ndim == 3 and out.shape[0] == 2:
            mask = out[1] - out[0]
        elif out.ndim == 2:
            mask = out
        else:
            mask = out.max(axis=0)
        # float logits first (linear resize preserves logit scale), then the
        # existing > 0.5 threshold at full resolution
        if mask.shape != (hgt, wid):
            mask = cv2.resize(mask.astype(np.float32), (wid, hgt), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)
        area_frac = float(mask.sum()) / max((box["w"] * box["h"]), 1)
        return mask, round(area_frac, 4)


class UltralyticsSegmenter:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # optional

        self.model = YOLO(cfg["path"])

    def segment(self, gray: np.ndarray, box: dict) -> tuple[np.ndarray, float]:
        res = self.model.predict(gray, verbose=False)[0]
        hgt, wid = gray.shape
        mask = np.zeros((hgt, wid), np.uint8)
        if res.masks is not None:
            m = res.masks.data.cpu().numpy()
            mask = (m.max(axis=0) > 0.5).astype(np.uint8)
            if mask.shape != (hgt, wid):
                import cv2

                mask = cv2.resize(mask, (wid, hgt), interpolation=cv2.INTER_NEAREST)
        area_frac = float(mask.sum()) / max(box["w"] * box["h"], 1)
        return mask, round(area_frac, 4)


class HeuristicSegmenter:
    name = "heuristic"

    def segment(self, gray: np.ndarray, box: dict) -> tuple[np.ndarray, float]:
        return segment_heuristic(gray, box)


def get_segmentation_backend(registry: dict | None = None) -> tuple[Any, str, str]:
    reg = registry or load_registry()
    cfg, backend, warning = resolve(reg, "segmentation")
    if backend == "onnx":
        return ONNXSegmenter(cfg), "onnx", warning
    if backend == "ultralytics":
        try:
            return UltralyticsSegmenter(cfg), "ultralytics", warning
        except ImportError:
            log.warning("ultralytics not installed — heuristic fallback")
            return HeuristicSegmenter(), "heuristic", "ultralytics unavailable; heuristic fallback"
    return HeuristicSegmenter(), "heuristic", warning
