"""Label-purity guards (semantic review v3 — docs/dataset-cards.md).

These tests make the scientific-review decisions *executable*: any future
converter or packaging change that reintroduces an invalid mapping
(Tire→cylinder, human→debris, Chain→net_like, …) fails the suite here,
before it can reach a training run.

Invariants enforced:

1. The supervised class list is exactly ``[wreck, debris, structure, tire]``
   and is identical across converters, bundle packager, config and registry.
2. ``cylinder`` / ``pipe`` / ``ghost_net`` are never supervised classes and
   never appear in any generated training manifest or YOLO label.
3. SCTD ``human`` boxes are excluded from detector supervision entirely.
4. MD-FLS ``Chain`` boxes are excluded (hard-negative only) and ``Tire``
   maps to ``tire`` — never ``cylinder``.
5. classes.txt in every YOLO export matches the registry class order exactly
   (an id-name mismatch silently corrupts every trained detection).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

EXPECTED_CLASSES = ["wreck", "debris", "structure", "tire"]
FORBIDDEN = {"cylinder", "pipe", "ghost_net", "net_like"}

# Approved v3 taxonomy — the ONLY detector class names allowed at runtime
# and in the API contract (see docs/dataset-cards.md, semantic review v3).
APPROVED_CLASSES = ["wreck", "debris", "structure", "tire"]

TRAINING = ROOT / "data" / "training"
SAMPLES = ROOT / "data" / "samples"


def _manifests() -> list[Path]:
    if not TRAINING.exists():
        pytest.skip("data/training not generated yet (run convert_datasets.py)")
    return sorted(TRAINING.glob("*/manifest.json"))


# ----------------------------------------------------------------- 1. taxonomy
def test_converters_use_expected_taxonomy() -> None:
    sys.path.insert(0, str(BACKEND / "scripts"))
    import convert_datasets

    assert convert_datasets.CLASSES == EXPECTED_CLASSES


def test_packager_uses_expected_taxonomy() -> None:
    training_dir = ROOT / "training"
    if not (training_dir / "package_for_colab.py").exists():
        pytest.skip("training/ packaging not present")
    sys.path.insert(0, str(training_dir))
    try:
        import package_for_colab
    finally:
        sys.path.remove(str(training_dir))
    assert package_for_colab.CLASSES == EXPECTED_CLASSES


def test_config_unsupported_classes_and_order() -> None:
    cfg_path = ROOT / "training" / "config.yaml"
    if not cfg_path.exists():
        pytest.skip("training/config.yaml not present")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    det = cfg["detection"]
    assert det["classes"] == EXPECTED_CLASSES
    for cls in FORBIDDEN:
        assert cls in det["unsupported_classes"], f"{cls} must be listed as unsupported"


def test_registry_class_order_matches_taxonomy() -> None:
    reg_path = ROOT / "models" / "registry.json"
    if not reg_path.exists():
        pytest.skip("models/registry.json not present")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert reg["detection"]["classes"] == EXPECTED_CLASSES


# --------------------------------------------------- 2. no forbidden classes
def test_no_forbidden_class_in_training_manifests() -> None:
    for mf in _manifests():
        data = json.loads(mf.read_text(encoding="utf-8"))
        for g in data.get("ground_truth", []):
            for b in g.get("boxes", []):
                assert b["class"] not in FORBIDDEN, (mf.name, b)


def test_no_forbidden_class_in_yolo_labels() -> None:
    """Class ids in YOLO label files must map only to the allowed taxonomy."""
    n_labels = 0
    for ydir in sorted(TRAINING.glob("*/yolo")) if TRAINING.exists() else []:
        classes = (ydir / "classes.txt").read_text(encoding="utf-8").split()
        assert classes == EXPECTED_CLASSES, ydir
        allowed_ids = set(range(len(classes)))
        for lbl in (ydir / "labels").glob("*.txt"):
            n_labels += 1
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cid = int(line.split()[0])
                assert cid in allowed_ids, (lbl, line)
                assert classes[cid] not in FORBIDDEN, (lbl, line)
    if n_labels == 0:
        pytest.skip("no YOLO exports generated yet")


# ----------------------------------------------------- 3. human exclusion
def test_sctd_human_boxes_excluded() -> None:
    mf = TRAINING / "sctd" / "manifest.json"
    if not mf.exists():
        pytest.skip("sctd manifest not generated")
    data = json.loads(mf.read_text(encoding="utf-8"))
    boxes = [b for g in data["ground_truth"] for b in g.get("boxes", [])]
    humans = [b for b in boxes if b.get("raw_class") == "human"]
    assert humans == [], "human boxes must be excluded from detector supervision"
    raws = {b.get("raw_class") for b in boxes}
    assert raws <= {"ship", "aircraft"}, raws
    assert boxes, "sctd manifest unexpectedly empty"


# --------------------------------------- 4. MD-FLS alias decisions (samples)
def _sample_manifest(name: str) -> dict:
    p = SAMPLES / name
    if not p.exists():
        pytest.skip(f"{name} not generated (run prepare_sample_data.py)")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fname", ["md_fls_watertank.json", "md_fls_watertank_demo.json"])
def test_mdfls_sample_manifest_purity(fname: str) -> None:
    data = _sample_manifest(fname)
    boxes = [b for g in data["ground_truth"] for b in g.get("boxes", [])]
    assert boxes, f"{fname} has no boxes"
    for b in boxes:
        assert b["class"] not in FORBIDDEN, b
        assert b.get("raw_class") != "Chain", "Chain is a hard negative — must not supervise"
        if b.get("raw_class") == "Tire":
            assert b["class"] == "tire", "Tire must map to tire, never cylinder"


# ----------------------------------------- 5. synthetic stays demo-only
def test_synthetic_classes_never_enter_training_data() -> None:
    """pipe/ghost_net may exist ONLY in the synthetic demo manifest."""
    synth = SAMPLES / "synthetic_sss.json"
    if not synth.exists():
        pytest.skip("synthetic_sss.json not generated")
    data = json.loads(synth.read_text(encoding="utf-8"))
    assert data.get("synthetic") is True, "synthetic manifest must be flagged synthetic"
    for mf in _manifests():
        data = json.loads(mf.read_text(encoding="utf-8"))
        assert data.get("synthetic") is not True, (
            f"{mf} is flagged synthetic — synthetic data must never enter supervised training"
        )


# ----------------------------------- 6. runtime: detector backends + priority


def _app_modules():
    """Import the runtime pipeline modules (backend package on sys.path)."""
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from app.pipeline import classify_natural as cn
    from app.pipeline import fusion, priority
    from app.pipeline.detect import HeuristicDetector, ONNXDetector, UltralyticsDetector

    return cn, fusion, priority, HeuristicDetector, ONNXDetector, UltralyticsDetector


def test_priority_tables_use_only_approved_classes() -> None:
    _cn, _fusion, priority, *_ = _app_modules()
    for table in (priority.TYPE_RISK, priority.ENTANGLEMENT, priority.ENVIRONMENTAL):
        assert set(table) <= {*APPROVED_CLASSES, "unknown_anomaly", "unknown"}, table.keys()
        assert FORBIDDEN.isdisjoint(table), f"forbidden class in priority table: {table.keys() & FORBIDDEN}"


def test_runtime_modules_have_no_net_like_relabelling() -> None:
    """The pipeline must never rewrite a detection's class to a forbidden name."""
    import re

    sources = {
        "runner": BACKEND / "app" / "pipeline" / "runner.py",
        "classify_natural": BACKEND / "app" / "pipeline" / "classify_natural.py",
        "detect": BACKEND / "app" / "pipeline" / "detect.py",
        "analysis_service": BACKEND / "app" / "analysis_service.py",
    }
    for mod_name, path in sources.items():
        src = path.read_text(encoding="utf-8")
        for bad in FORBIDDEN:
            assert not re.search(rf'["\']{bad}["\']\s*\]', src), f"{mod_name}: assignment of forbidden class {bad}"


def test_heuristic_detector_emits_only_approved_classes() -> None:
    import numpy as np

    _cn, _f, _p, HeuristicDetector, *_ = _app_modules()
    rng = np.random.default_rng(3)
    img = rng.normal(128, 12, (240, 320)).astype(np.uint8)
    dets = HeuristicDetector().detect(img, {})
    for d in dets:
        assert d["class"] in set(APPROVED_CLASSES), d


def test_onnx_detector_class_list_validated() -> None:
    _cn, _f, _p, _H, ONNXDetector, _U = _app_modules()
    det = ONNXDetector.__new__(ONNXDetector)
    det.classes = ["wreck", "debris", "structure", "tire"]
    det.nc = 4
    det.conf = 0.5
    det.iou = 0.45
    det.fmt = "v8"
    import numpy as np

    nc = 4
    n = 20
    rng = np.random.default_rng(0)
    tensor = rng.random((1, 4 + nc, n)).astype(np.float32)
    tensor[0, :4, :] *= 320  # plausible pixel coords
    tensor[0, 4:8, :] = 0.0
    tensor[0, 4:8, 3] = 0.9  # one confident detection
    out = det._parse([tensor], nc, det.conf, det.iou)
    assert len(out) >= 1
    assert all(0 <= c < nc for _xy, c, _s in out)
