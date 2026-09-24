"""Pluggable object detection.

Backends:
- ``onnx``        — generic YOLOv8/v11 ONNX inference (CPU via onnxruntime)
- ``ultralytics`` — ultralytics YOLO with .pt weights (optional install)
- ``heuristic``   — deterministic sonar baseline (fallback, always works)

The ONNX path implements letterboxing and NMS itself, so any standard
YOLOv8/v11 export (single tensor, multi-output, or end2end) works without
the ultralytics dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..fallback.candidates import detect_heuristic
from ..models.registry import load_registry, resolve

log = logging.getLogger("aqua.detect")


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, int, int]]:
    import cv2

    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size), 114, dtype=np.uint8)
    canvas[pady : pady + nh, padx : padx + nw] = resized
    return canvas, (scale, padx, pady)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Pure-numpy NMS over xyxy boxes. Returns kept indices (desc score)."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=int)


class ONNXDetector:
    def __init__(self, cfg: dict):
        import onnxruntime as ort

        self.session = ort.InferenceSession(cfg["path"], providers=["CPUExecutionProvider"])
        self.classes: list[str] = cfg.get("classes") or ["debris"]
        self.nc = len(self.classes)
        self.input_size = int(cfg.get("input_size", 640))
        self.conf = float(cfg.get("conf_threshold", 0.15))
        self.iou = float(cfg.get("nms_iou", 0.45))
        self.fmt = cfg.get("format", "v8")

    def detect(self, gray: np.ndarray, meta: dict | None = None) -> list[dict]:

        h, w = gray.shape[:2]
        inp, (scale, padx, pady) = letterbox(gray, self.input_size)
        # Sonar imagery has no color: replicate the (letterboxed, /255-ated)
        # grayscale plane to 3 channels — no RGB information is invented.
        blob = np.repeat(inp[None, None, :, :], 3, axis=1).astype(np.float32) / 255.0
        out = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        dets = self._parse(out, self.nc, self.conf, self.iou)
        results = []
        for box_xyxy, cls_id, score in dets:
            x1, y1, x2, y2 = (np.asarray(box_xyxy) - np.array([padx, pady, padx, pady])) / scale
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cls_name = self.classes[cls_id] if 0 <= cls_id < len(self.classes) else "debris"
            results.append(
                {
                    "box": {"x": float(x1), "y": float(y1), "w": float(x2 - x1), "h": float(y2 - y1)},
                    "class": cls_name,
                    "score": round(float(score), 3),
                }
            )
        return results

    def _parse(self, outputs, nc, conf, iou):
        def to_xyxy(cx, cy, bw, bh):
            return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]

        if self.fmt == "end2end" or (
            isinstance(outputs, (list, tuple)) and outputs[0].ndim == 3 and outputs[0].shape[2] == 6
        ):
            t = np.asarray(outputs[0])[0]
            keep = t[:, 5] >= conf
            t = t[keep]
            return [(t[i, :4], int(t[i, 4]), t[i, 5]) for i in range(len(t))]

        if self.fmt == "v8_multi":
            boxes_t = np.asarray(outputs[0])[0]  # (4, N)
            cls_t = np.asarray(outputs[1])[0]  # (nc, N)
            scores = cls_t.max(axis=0)
            cls_ids = cls_t.argmax(axis=0)
            cand = np.where(scores >= conf)[0]
            if len(cand) == 0:
                return []
            xyxy = np.array([to_xyxy(*boxes_t[:4, i]) for i in cand])
            keep = nms(xyxy, scores[cand], iou)
            return [(xyxy[i], int(cls_ids[cand[i]]), scores[cand[i]]) for i in keep]

        # single tensor: (1, 4+nc, N) or (1, N, 4+nc)
        t = np.asarray(outputs[0])
        if t.ndim == 3 and t.shape[1] == 4 + nc:
            t = t[0].T  # (N, 4+nc)
        elif t.ndim == 3 and t.shape[2] == 4 + nc:
            t = t[0]
        else:
            raise ValueError(f"unsupported YOLO output shape {t.shape}")
        scores = t[:, 4 : 4 + nc].max(axis=1)
        cls_ids = t[:, 4 : 4 + nc].argmax(axis=1)
        cand = np.where(scores >= conf)[0]
        if len(cand) == 0:
            return []
        xyxy = np.array([to_xyxy(*t[i, :4]) for i in cand])
        keep = nms(xyxy, scores[cand], iou)
        return [(xyxy[i], int(cls_ids[cand[i]]), scores[cand[i]]) for i in keep]


class UltralyticsDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # optional dependency

        self.model = YOLO(cfg["path"])
        self.classes: list[str] = cfg.get("classes") or list(self.model.names.values())
        self.conf = float(cfg.get("conf_threshold", 0.15))
        self.iou = float(cfg.get("nms_iou", 0.45))

    def detect(self, gray: np.ndarray, meta: dict | None = None) -> list[dict]:
        res = self.model.predict(gray, conf=self.conf, iou=self.iou, verbose=False)[0]
        results = []
        for box, cls_id, score in zip(
            res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), strict=True
        ):
            x1, y1, x2, y2 = map(float, box)
            cls_name = self.classes[int(cls_id)] if int(cls_id) < len(self.classes) else "debris"
            results.append(
                {
                    "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "class": cls_name,
                    "score": round(float(score), 3),
                }
            )
        return results


class HeuristicDetector:
    name = "heuristic"

    def detect(self, gray: np.ndarray, meta: dict | None = None) -> list[dict]:
        return detect_heuristic(gray, meta or {})


def get_detection_backend(registry: dict | None = None) -> tuple[Any, str, str]:
    reg = registry or load_registry()
    cfg, backend, warning = resolve(reg, "detection")
    if backend == "onnx":
        return ONNXDetector(cfg), "onnx", warning
    if backend == "ultralytics":
        try:
            return UltralyticsDetector(cfg), "ultralytics", warning
        except ImportError:
            log.warning("ultralytics not installed — heuristic fallback")
            return HeuristicDetector(), "heuristic", "ultralytics unavailable; heuristic fallback"
    return HeuristicDetector(), "heuristic", warning
