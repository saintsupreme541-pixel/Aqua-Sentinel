"""Natural-vs-artificial classification.

Returns ``p_artificial`` — the probability the detected object is man-made
(not a rock / sand formation).  The heuristic baseline is explainable
(intensity homogeneity, contrast, convexity, contour straightness); a
trained CNN replaces it via the model registry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fallback.classifier_heuristic import classify_heuristic
from ..models.registry import load_registry, resolve


class ONNXClassifier:
    def __init__(self, cfg: dict):
        import onnxruntime as ort

        self.session = ort.InferenceSession(cfg["path"], providers=["CPUExecutionProvider"])

    def classify(
        self, gray: np.ndarray, box: dict, mask: np.ndarray, class_name: str | None = None
    ) -> tuple[float, dict]:
        import cv2

        x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
        x1, y1 = min(gray.shape[1], int(box["x"] + box["w"])), min(gray.shape[0], int(box["y"] + box["h"]))
        crop = gray[y0:y1, x0:x1]
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
        blob = crop[None, None, :, :].astype(np.float32) / 255.0
        out = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0]
        arr = np.asarray(out).reshape(-1)
        # Two-class LOGITS [natural, artificial] — softmax, not sigmoid.
        if arr.size >= 2:
            e = np.exp(arr - arr.max())  # numerically stable
            p = float(e[1] / e.sum())
        else:
            p = float(arr[0])
        return round(float(np.clip(p, 0.05, 0.95)), 3), {"backend": "onnx"}


class HeuristicClassifier:
    name = "heuristic"

    def classify(
        self, gray: np.ndarray, box: dict, mask: np.ndarray, class_name: str | None = None
    ) -> tuple[float, dict]:
        p, features = classify_heuristic(gray, box, mask)
        del class_name  # no class-based priors under the approved taxonomy
        return p, features


def get_classifier_backend(registry: dict | None = None) -> tuple[Any, str, str]:
    reg = registry or load_registry()
    cfg, backend, warning = resolve(reg, "classifier")
    if backend == "onnx":
        try:
            return ONNXClassifier(cfg), "onnx", warning
        except Exception as e:  # pragma: no cover
            return HeuristicClassifier(), "heuristic", f"classifier onnx failed ({e}); heuristic fallback"
    return HeuristicClassifier(), "heuristic", warning
