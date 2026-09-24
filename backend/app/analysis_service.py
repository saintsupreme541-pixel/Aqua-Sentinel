"""Single-image (and small-sequence) analysis service.

Runs the full multi-stage pipeline **synchronously and in-memory** for the
``/api/analyze`` endpoint — the same stage modules the survey job runner
uses (quality → preprocess → detect → segment → classify → shadow →
physics → anomaly → fusion), but without DB persistence.  With multiple
frames, a lightweight appearance-consistency check across frames provides
the temporal-consistency signal; with a single frame it is honestly
reported as unavailable.  Metadata (GPS/heading/altitude/range) is
optional and never fabricated.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import cv2
import numpy as np

from . import association
from .config import settings
from .models.registry import load_registry
from .pipeline import anomaly as anomaly_mod
from .pipeline import classify_natural, detect, fusion, preprocess, runner
from .pipeline import dimensions as dimensions_mod
from .pipeline import physics as physics_mod
from .pipeline import priority as priority_mod
from .pipeline import quality as quality_mod
from .pipeline import segment as segment_mod
from .pipeline import shadow as shadow_mod
from .pipeline.geolocate import locate_detection

# ---- singletons: models load once per process, not per request -------------
_REGISTRY: dict[str, Any] | None = None
_BACKENDS: dict[str, tuple[Any, str, str]] = {}


def _get_backends() -> tuple[dict[str, Any], dict[str, str], list[str]]:
    global _REGISTRY, _BACKENDS
    if _BACKENDS is None or not _BACKENDS:
        _REGISTRY = load_registry()
        det, det_b, det_w = detect.get_detection_backend(_REGISTRY)
        seg, seg_b, seg_w = segment_mod.get_segmentation_backend(_REGISTRY)
        cls, cls_b, cls_w = classify_natural.get_classifier_backend(_REGISTRY)
        _BACKENDS = {
            "detection": (det, det_b, det_w),
            "segmentation": (seg, seg_b, seg_w),
            "classifier": (cls, cls_b, cls_w),
        }
    backends = {k: v[1] for k, v in _BACKENDS.items()}
    notes = [v[2] for v in _BACKENDS.values() if v[2]]
    if all(b == "heuristic" for b in backends.values()):
        notes.append(
            "no trained weights present — running the deterministic sonar-heuristic baseline "
            "(clearly labeled, not presented as trained AI)"
        )
    return {k: v[0] for k, v in _BACKENDS.items()}, backends, notes


def analyze_images(
    images: list[tuple[str, bytes]],
    meta: dict[str, Any],
    preset: str = "light",
) -> dict[str, Any]:
    """Analyze 1..N sonar frames synchronously; return the spec §17 payload."""
    analysis_id = f"AQ-{uuid.uuid4().hex[:12].upper()}"
    t_start = time.perf_counter()

    decoded: list[tuple[str, np.ndarray]] = []
    for filename, content in images:
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"'{filename}' is not a decodable image")
        decoded.append((filename, img))

    models, backends, notes = _get_backends()
    detector, segmenter, classifier = models["detection"], models["segmentation"], models["classifier"]

    all_dets: list[dict[str, Any]] = []
    image_blocks: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    for idx, (filename, gray) in enumerate(decoded):
        imeta = {**meta}
        base = time.perf_counter()

        quality = quality_mod.assess(gray, imeta)
        processed, params = preprocess.preprocess(gray, imeta, preset=preset)
        timings[f"img{idx}"] = round((time.perf_counter() - base) * 1000, 1)

        raw_dets = detector.detect(processed, imeta)
        det_ids = [f"det_{idx}_{k}" for k in range(len(raw_dets))]

        masks: dict[str, np.ndarray] = {}
        for d, did in zip(raw_dets, det_ids, strict=True):
            mask, frac = segmenter.segment(processed, d["box"])
            masks[did] = mask
            d["mask_area_frac"] = frac

        for d, did in zip(raw_dets, det_ids, strict=True):
            d["id"] = did
            b = d["box"]
            p_art, feats = classifier.classify(processed, b, masks[did], d.get("class"))
            d["p_artificial"] = p_art
            d["classifier_features"] = feats

            direction = shadow_mod.default_direction(imeta)
            info, evidence = shadow_mod.analyze_shadow(
                processed,
                b,
                direction=direction,
                range_m=imeta.get("range_m"),
                altitude_m=imeta.get("altitude_m"),
                image_height=processed.shape[0],
            )
            d["shadow"] = info.model_dump()
            d["shadow_evidence"] = evidence

            d["physics"] = physics_mod.analyze_physics(
                box=b, shadow=d["shadow"], meta=imeta, image_height=processed.shape[0]
            ).model_dump()

            box_with_id = {**b, "id": did}
            anom_scores, _ = anomaly_mod.anomaly_scores(processed, [box_with_id])
            d["anomaly_score"] = float(anom_scores.get(did, 0.5))

            geo = locate_detection(
                meta=imeta,
                slant_range_m=None,
                image_height=processed.shape[0],
                row=b["y"] + b["h"] / 2,
                reference_frame_id=filename,
            )
            dims = dimensions_mod.estimate_dimensions(
                box=b,
                image_height=processed.shape[0],
                range_m=imeta.get("range_m"),
                altitude_m=imeta.get("altitude_m"),
                height_estimate_m=d["shadow"].get("height_estimate_m"),
            )
            d["geolocation"] = geo
            d["dimensions"] = dims.model_dump()
            d["image_index"] = idx
            d["image_filename"] = filename

        all_dets.extend(raw_dets)

        image_blocks.append(
            {
                "filename": filename,
                "width": int(processed.shape[1]),
                "height": int(processed.shape[0]),
                "quality": quality.model_dump(),
                "preprocess_params": params,
                "overlay_url": None,  # filled by the route after persistence
            }
        )

    # ---- survey-level: consistency + fusion + status + priority -------------
    # Phase 2: ONE canonical consistency implementation (app.association) is
    # shared with the survey runner — no divergent per-service algorithm.
    cons = association.compute_consistency(all_dets) if len(decoded) > 1 else {}
    for d in all_dets:
        c = cons.get(d["id"], {})
        d["evidence_signals"] = {
            "detection": float(d["score"]),
            "segmentation": _segmentation_evidence(d.get("mask_area_frac")),
            "natural": float(d["p_artificial"]),
            "shadow": d["shadow_evidence"],
            "physics": (d["physics"].get("geometry_score") if d["physics"].get("metadata_sufficient") else None),
            "consistency": c.get("evidence"),
        }
        d["evidence_availability"] = {
            "detection": True,
            "segmentation": True,
            "natural": True,
            "shadow": d["shadow_evidence"] is not None,
            "physics": d["physics"].get("metadata_sufficient") is True,
            "consistency": c.get("availability", False),
        }
        conf, breakdown = fusion.fuse(
            d["evidence_signals"],
            d["evidence_availability"],
            gain=settings.fusion_gain,
            intercept=settings.fusion_intercept,
        )
        d["final_confidence"] = conf
        d["evidence_breakdown"] = breakdown
        d["status"] = fusion.decide_status(
            conf,
            d.get("anomaly_score"),
            d["score"],
            confirm=settings.confirm_threshold,
            review=settings.review_threshold,
        )

        prio = priority_mod.compute_priority(
            class_name=d["class"],
            fusion_confidence=conf,
            width_m=d["dimensions"].get("width_m"),
            height_m=d["dimensions"].get("height_m"),
            dimensions_estimable=d["dimensions"].get("estimable", False),
        )
        d["priority"] = prio.model_dump()

    t_total = round((time.perf_counter() - t_start) * 1000, 1)
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "image": image_blocks[0] if image_blocks else None,
        "images": image_blocks,
        "detections": [_public_det(d) for d in all_dets],
        "backends": backends,
        "backend_notes": notes,
        "timings_ms": {**timings, "total": t_total},
    }


def _segmentation_evidence(mask_area_frac: float | None) -> float:
    """Same bell-shaped agreement as the survey runner (runner.py)."""
    return runner.segmentation_evidence(mask_area_frac)


def _public_det(d: dict[str, Any]) -> dict[str, Any]:
    """Shape the internal detection dict into the spec §17 response block."""
    shadow = d.get("shadow") or {}
    phys = d.get("physics") or {}
    geo = d.get("geolocation") or {}
    dims = d.get("dimensions") or {}
    return {
        "id": d.get("id") or f"det_{uuid.uuid4().hex[:8]}",
        "class": d.get("class", "debris"),
        "bbox": [
            round(float(d["box"]["x"]), 1),
            round(float(d["box"]["y"]), 1),
            round(float(d["box"]["w"]), 1),
            round(float(d["box"]["h"]), 1),
        ],
        "yolo_confidence": round(float(d["score"]), 3),
        "segmentation_score": round(_segmentation_evidence(d.get("mask_area_frac")), 3),
        "artificial_probability": round(float(d.get("p_artificial", 0.5)), 3),
        "shadow": {
            "detected": bool(shadow.get("valid")),
            "length_px": shadow.get("length_px"),
            "length_m": shadow.get("length_m"),
            "area_px": shadow.get("area_px"),
            "angle_deg": shadow.get("angle_deg"),
            "score": d.get("shadow_evidence"),
            "note": shadow.get("note", ""),
        },
        "physics": {
            "metadata_sufficient": phys.get("metadata_sufficient", False),
            "estimated_height_m": phys.get("estimated_height_m"),
            "grazing_angle_deg": phys.get("grazing_angle_deg"),
            "geometry_score": phys.get("geometry_score"),
            "note": phys.get("note", ""),
        },
        "temporal_consistency": d["evidence_signals"]["consistency"],
        "final_confidence": d["final_confidence"],
        "decision": d["status"],
        "geolocation": {
            "known": geo.get("known", False),
            "status": geo.get("status", "approximate" if geo.get("known") else "unknown"),
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
            "uncertainty_m": geo.get("uncertainty_m"),
            "provenance": geo.get("provenance"),
            "note": geo.get("note", ""),
        },
        "dimensions": dims,
        "priority": d.get("priority", {}),
    }
