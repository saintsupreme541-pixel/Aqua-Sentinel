"""Model registry.

Reads ``models/registry.json`` and resolves each task to a concrete
backend.  A configured backend whose weights are missing **falls back to
the deterministic heuristic baseline** and records that fact — the UI and
reports show which backend produced every result (no silent claims of "AI"
when a heuristic ran).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import settings

log = logging.getLogger("aqua.registry")

DEFAULT_REGISTRY: dict[str, Any] = {
    "detection": {
        "backend": "heuristic",
        "path": "",
        "classes": ["debris"],
        "input_size": 640,
        "conf_threshold": 0.15,
        "nms_iou": 0.45,
        "format": "v8",
    },
    "segmentation": {"backend": "heuristic", "path": ""},
    "classifier": {"backend": "heuristic", "path": ""},
}


def load_registry(path: Path | None = None) -> dict[str, Any]:
    p = path or settings.model_registry_path
    if not p.exists():
        log.warning("model registry %s not found — using heuristic defaults", p)
        return json.loads(json.dumps(DEFAULT_REGISTRY))
    data = json.loads(p.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_REGISTRY)
    for task, cfg in data.items():
        if task in merged and isinstance(cfg, dict):
            merged[task].update(cfg)
    return merged


# Repo root, derived from this file's location — NOT the process CWD, so a
# relative registry path like ``models/weights/yolo-sss.onnx`` resolves the
# same way whether uvicorn is started from backend/ or the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # backend/app/models/registry.py → repo root


def _weight_path(weight_path: str) -> Path:
    """Resolve a registry weight path: absolute as-is, else repo-root-relative."""
    p = Path(weight_path)
    return p if p.is_absolute() else _REPO_ROOT / p


def resolve(registry: dict[str, Any], task: str) -> tuple[dict[str, Any], str, str]:
    """Returns (config, backend_name, warning). backend_name is the *actual* backend used."""
    cfg = registry.get(task, DEFAULT_REGISTRY.get(task, {}))
    backend = cfg.get("backend", "heuristic")
    weight_path = cfg.get("path") or ""
    if backend in ("onnx", "ultralytics") and weight_path:
        p = _weight_path(weight_path)
        if not p.exists():
            log.warning("%s backend configured but weights missing (%s) — heuristic fallback", task, p)
            cfg = {**cfg, "backend": "heuristic"}
            return cfg, "heuristic", f"{task}: configured {backend} weights missing; heuristic fallback"
        if str(p) != weight_path:  # relative path → hand adapters the absolute one
            cfg = {**cfg, "path": str(p)}
    if backend == "heuristic":
        return cfg, "heuristic", ""
    return cfg, backend, ""
