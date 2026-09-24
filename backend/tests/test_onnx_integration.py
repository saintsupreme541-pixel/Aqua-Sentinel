"""Integration tests for the trained ONNX weights (models/weights/).

Verifies the REAL runtime contract — not just registry.json:

- the registry resolves all three tasks to the ``onnx`` backend with
  existing, CWD-independent absolute paths;
- the unified YOLOv8n detector runs end-to-end on a sonar image through
  the app adapter (3-channel replicated input, [1,8,8400] parse);
- the U-Net segmenter handles a non-512 sonar resolution (static
  [1,1,512,512] model, resize in / logits back out);
- the classifier emits a proper softmax probability (p[natural] +
  p[artificial] ~= 1).

Skips cleanly (so CI without weights stays green) only when onnxruntime or
the weight files are genuinely absent; a registry misconfiguration or an
adapter crash is a FAILURE, not a skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent

WEIGHTS = ROOT / "models" / "weights"
DETECT_ONNX = WEIGHTS / "yolo-sss.onnx"
SEG_ONNX = WEIGHTS / "unet-sss.onnx"
CLS_ONNX = WEIGHTS / "natart-sss.onnx"

_requires_ort = pytest.importorskip("onnxruntime")
if not (DETECT_ONNX.exists() and SEG_ONNX.exists() and CLS_ONNX.exists()):
    pytest.skip("trained ONNX weights not present in models/weights/", allow_module_level=True)

APPROVED_CLASSES = {"wreck", "debris", "structure", "tire"}


def _sample_sonar(shape: tuple[int, int] = (522, 862)) -> np.ndarray:
    """A synthetic sonar-like scene: bright blob + dark shadow + noise."""
    import cv2

    rng = np.random.default_rng(7)
    img = rng.normal(90, 18, shape).astype(np.uint8)
    cv2.circle(img, (430, 200), 38, 235, -1)
    cv2.ellipse(img, (455, 250), (60, 18), 0, 0, 360, 15, -1)  # shadow-like dark region
    return img


@pytest.fixture(scope="module")
def registry() -> dict:
    from app.models.registry import load_registry

    return load_registry()


# --------------------------------------------------------------- registry ---
def test_registry_resolves_all_onnx_backends(registry: dict) -> None:
    from app.models.registry import resolve

    for task in ("detection", "segmentation", "classifier"):
        cfg, backend, warning = resolve(registry, task)
        assert backend == "onnx", (task, cfg, warning)
        assert warning == "", (task, warning)
        assert Path(cfg["path"]).is_absolute(), cfg
        assert Path(cfg["path"]).exists(), cfg


def test_registry_paths_independent_of_cwd(registry: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative registry paths must resolve via repo root, not the CWD."""
    from app.models.registry import load_registry, resolve

    monkeypatch.chdir(tmp_path)  # any unrelated CWD
    fresh = load_registry()  # reload under the new CWD
    for task in ("detection", "segmentation", "classifier"):
        cfg, backend, _ = resolve(fresh, task)
        assert backend == "onnx", task
        assert Path(cfg["path"]).exists(), cfg


# --------------------------------------------------------------- detector ---
def test_detector_loads_and_reports_onnx(registry: dict) -> None:
    from app.pipeline.detect import get_detection_backend

    det, backend, warning = get_detection_backend(registry)
    assert backend == "onnx"
    assert warning == ""
    assert det.classes == ["wreck", "debris", "structure", "tire"]
    assert det.input_size == 640
    assert det.conf == pytest.approx(0.15)
    assert det.iou == pytest.approx(0.45)
    assert det.fmt == "v8"


def test_detector_real_sonar_inference(registry: dict) -> None:
    from app.pipeline.detect import get_detection_backend

    det, backend, _ = get_detection_backend(registry)
    img = _sample_sonar()
    dets = det.detect(img, {})
    assert isinstance(dets, list)
    for d in dets:
        assert d["class"] in APPROVED_CLASSES, d
        assert 0.0 <= d["score"] <= 1.0, d
        b = d["box"]
        assert 0 <= b["x"] < 862 and 0 <= b["y"] < 522, d
        assert b["w"] > 0 and b["h"] > 0, d


# -------------------------------------------------------------- segmenter ---
def test_segmenter_non512_resolution(registry: dict) -> None:
    from app.pipeline.segment import get_segmentation_backend

    seg, backend, warning = get_segmentation_backend(registry)
    assert backend == "onnx"
    assert warning == ""
    img = _sample_sonar()  # 522x862 — NOT the model's native 512x512
    box = {"x": 380.0, "y": 150.0, "w": 120.0, "h": 120.0}
    mask, frac = seg.segment(img, box)
    assert mask.shape == img.shape, "mask must be returned at original resolution"
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1}
    assert 0.0 <= frac <= 1.0


def test_segmenter_odd_small_resolution(registry: dict) -> None:
    """An off-size small image (smaller than 512) must also work."""
    from app.pipeline.segment import get_segmentation_backend

    seg, _b, _w = get_segmentation_backend(registry)
    img = _sample_sonar(shape=(197, 233))
    box = {"x": 90.0, "y": 60.0, "w": 60.0, "h": 60.0}
    mask, _frac = seg.segment(img, box)
    assert mask.shape == img.shape


# -------------------------------------------------------------- classifier --
def test_classifier_softmax_probabilities(registry: dict) -> None:
    from app.pipeline.classify_natural import get_classifier_backend

    cls, backend, warning = get_classifier_backend(registry)
    assert backend == "onnx"
    assert warning == ""
    img = _sample_sonar()
    box = {"x": 380.0, "y": 150.0, "w": 120.0, "h": 120.0}
    p_art, feats = cls.classify(img, box, np.zeros(img.shape, np.uint8), "debris")
    assert feats.get("backend") == "onnx"
    assert 0.05 <= p_art <= 0.95  # existing clip
    # softmax sanity: the two probabilities sum to ~1
    import cv2

    crop = cv2.resize(img[150:270, 380:500], (64, 64), interpolation=cv2.INTER_LINEAR)
    blob = crop[None, None, :, :].astype(np.float32) / 255.0
    logits = cls.session.run(None, {cls.session.get_inputs()[0].name: blob})[0]
    arr = np.asarray(logits).reshape(-1)
    e = np.exp(arr - arr.max())
    probs = e / e.sum()
    assert probs.size == 2
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    # adapter contract = softmax at index 1 with the existing 0.05–0.95
    # fusion clip (a saturated model can legitimately hit the clip bound)
    assert float(np.clip(probs[1], 0.05, 0.95)) == pytest.approx(p_art, abs=1e-3)


# ------------------------------------------------- live backend reporting ---
def test_analysis_service_reports_all_onnx(registry: dict) -> None:
    """The runtime backend map — what /api/analyze reports — must be onnx."""
    import app.analysis_service as asvc

    _models, backends, notes = asvc._get_backends()
    assert backends == {"detection": "onnx", "segmentation": "onnx", "classifier": "onnx"}, backends
    assert notes == [] or all("fallback" not in n.lower() for n in notes), notes
