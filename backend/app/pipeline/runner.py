"""Pipeline orchestration for a survey job.

Runs in a worker thread: per-image stages (quality → preprocess → detect →
segment → classify → shadow → anomaly → geolocate → dimensions), then the
survey-level pass (consistency → fusion → status → priority).  Emits SSE
events as stages complete so the dashboard can stream progress.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import cv2
import numpy as np

from .. import db
from .. import navigation as nav
from ..config import settings
from ..fallback.candidates import bright_sparse_frac
from ..models.registry import load_registry
from . import anomaly as anomaly_mod
from . import classify_natural, consistency, detect, fusion, geolocate, preprocess, segment
from . import dimensions as dimensions_mod
from . import physics as physics_mod
from . import priority as priority_mod
from . import quality as quality_mod
from . import shadow as shadow_mod

log = logging.getLogger("aqua.runner")


def segmentation_evidence(mask_area_frac: float | None) -> float:
    """Bell-shaped agreement between the detection box and its mask (0–1).

    Best when the mask covers 20–65% of the box.  A mask that barely covers
    the box is a weak catch; a mask that swallows the whole box is suspicious
    (the detector box and mask agree trivially).  Shared with the
    single-image service so both consumers score identically.
    """
    if mask_area_frac is None:
        return 0.5
    f = mask_area_frac
    if f < 0.03:
        return 0.1
    if f < 0.20:
        return 0.1 + 0.9 * (f - 0.03) / 0.17  # 0.1 → 1.0
    if f <= 0.65:
        return 1.0
    if f < 0.95:
        return 1.0 - 0.7 * (f - 0.65) / 0.30  # 1.0 → 0.3
    return 0.2


def _det_id(image_id: str, idx: int) -> str:
    return f"det_{image_id.split('_')[-1]}_{idx}"


def analyze_survey(survey_id: str, job_id: str, emit) -> None:
    t_start = time.perf_counter()
    survey = db.get_survey(survey_id)
    images = db.get_images(survey_id)
    if not survey or not images:
        db.update_job(job_id, status="failed", stage="init", message="survey or images missing", error="no images")
        return

    survey_meta: dict[str, Any] = json.loads(survey["meta_json"])
    preset = survey_meta.get("preprocess_preset", settings.preprocess_preset)
    n = len(images)

    # Geolocation v2: synchronize frames against the survey's genuine
    # uploaded navigation track (if any).  The track NEVER creates a fake
    # position — frames without a timestamp simply keep their own metadata.
    nav_track: list[dict[str, Any]] = []
    try:
        nav_track = db.get_nav_records(survey_id)
    except Exception:  # pragma: no cover — table exists from migration on
        nav_track = []
    for r in nav_track:
        r["_ts"] = nav._parse_ts(r.get("ts"))
    tow_cfg = nav.survey_towfish_config(survey_meta)

    db.update_job(
        job_id, status="running", stage="init", progress=0, message=f"Loading models — analyzing {n} image(s)"
    )

    registry = load_registry()
    detector, det_backend, det_warn = detect.get_detection_backend(registry)
    segmenter, seg_backend, seg_warn = segment.get_segmentation_backend(registry)
    classifier, cls_backend, cls_warn = classify_natural.get_classifier_backend(registry)
    backends = {"detection": det_backend, "segmentation": seg_backend, "classifier": cls_backend}
    warnings = [w for w in (det_warn, seg_warn, cls_warn) if w]

    timings: dict[str, float] = {}

    for i, img in enumerate(images):
        base = (i / n) * 100.0
        imeta: dict[str, Any] = {**survey_meta, **json.loads(img["meta_json"])}
        db.set_image_status(img["id"], "analyzing")
        emit(job_id, "quality", base, f"Quality assessment — {img['filename']}")
        t0 = time.perf_counter()
        gray = preprocess.load_grayscale(img["raw_path"])
        quality = quality_mod.assess(gray, imeta)
        timings["quality"] = (time.perf_counter() - t0) * 1000

        emit(job_id, "preprocess", base + 4, f"Sonar preprocessing — {img['filename']}")
        t0 = time.perf_counter()
        processed, params = preprocess.preprocess(gray, imeta, preset=preset)
        proc_rel = f"processed/{survey_id}/{img['id']}/processed.png"
        preprocess.save_preview(processed, settings.storage_dir / proc_rel)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000

        emit(job_id, "detect", base + 12, f"Object detection — {img['filename']}")
        t0 = time.perf_counter()
        raw_dets = detector.detect(processed, imeta)
        timings["detect"] = (time.perf_counter() - t0) * 1000
        det_ids = [_det_id(img["id"], k) for k in range(len(raw_dets))]

        if raw_dets:
            emit(job_id, "segment", base + 20, f"Segmentation — {len(raw_dets)} candidate(s)")
            t0 = time.perf_counter()
            masks: dict[str, np.ndarray] = {}
            mask_paths: dict[str, str] = {}
            for d, did in zip(raw_dets, det_ids, strict=True):
                mask, frac = segmenter.segment(processed, d["box"])
                rel = f"masks/{survey_id}/{img['id']}/mask_{did}.png"

                # *255 → {0, 255} PNG: black background, white mask (legible
                # in the Lab overlay and the PDF; {0,1} would render black)
                (settings.storage_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(settings.storage_dir / rel), mask * 255):
                    mask_paths[did] = rel
                else:  # pragma: no cover — disk/permission failure
                    log.warning("mask write failed: %s", rel)
                masks[did] = mask
                d["mask_area_frac"] = frac
            timings["segment"] = (time.perf_counter() - t0) * 1000

            emit(job_id, "classify", base + 26, "Natural vs artificial classification")
            t0 = time.perf_counter()
            for d, did in zip(raw_dets, det_ids, strict=True):
                b = d["box"]
                p_art, feats = classifier.classify(processed, b, masks[did], d.get("class"))
                sparse = bright_sparse_frac(processed, b)
                feats["sparse_bright_frac"] = round(sparse, 3)
                # Taxonomy note (semantic review v3): no supervised net class
                # exists, so sparse-bright filament texture is recorded as
                # *evidence only* (raises p_artificial modestly) — the class
                # label is never rewritten to net_like/ghost_net.  A high
                # anomaly score + this signature routes to human review.
                if 0.06 <= sparse <= 0.60:
                    p_art = min(0.95, p_art + 0.10)
                    feats["note"] = "sparse-bright filamentous texture — man-made signature (evidence only)"
                d["p_artificial"] = p_art
                d["classifier_features"] = feats
            timings["classify"] = (time.perf_counter() - t0) * 1000

            emit(job_id, "shadow", base + 32, "Acoustic shadow analysis")
            t0 = time.perf_counter()
            for d, _did in zip(raw_dets, det_ids, strict=True):
                direction = shadow_mod.default_direction(imeta)
                info, evidence = shadow_mod.analyze_shadow(
                    processed,
                    d["box"],
                    direction=direction,
                    range_m=imeta.get("range_m"),
                    altitude_m=imeta.get("altitude_m"),
                    image_height=processed.shape[0],
                )
                d["shadow"] = info.model_dump()
                d["shadow_evidence"] = evidence
            timings["shadow"] = (time.perf_counter() - t0) * 1000

            emit(job_id, "physics", base + 35, "Physics-informed shadow geometry")
            t0 = time.perf_counter()
            for d in raw_dets:
                d["physics"] = physics_mod.analyze_physics(
                    box=d["box"], shadow=d["shadow"], meta=imeta, image_height=processed.shape[0]
                ).model_dump()
            timings["physics"] = (time.perf_counter() - t0) * 1000

            emit(job_id, "anomaly", base + 38, "Unknown anomaly detection")
            t0 = time.perf_counter()
            boxes_for_anom = [{**d["box"], "id": did} for d, did in zip(raw_dets, det_ids, strict=True)]
            anom_scores, anom_meta = anomaly_mod.anomaly_scores(processed, boxes_for_anom)
            timings["anomaly"] = (time.perf_counter() - t0) * 1000
        else:
            masks, mask_paths, anom_scores, anom_meta = {}, {}, {}, {"trained": False}
        backends = {**backends, "anomaly": "autoencoder" if anom_meta.get("trained") else "heuristic"}

        emit(job_id, "geolocate", base + 46, "Geolocation & dimensions")
        t0 = time.perf_counter()
        # Frame timestamp sync (§4): match this frame's capture time to the
        # uploaded track — exact / linear_interpolation / nearest, recorded.
        frame_nav_sync: dict[str, Any] | None = None
        frame_ts = nav.frame_timestamp(img)
        if nav_track and frame_ts is not None:
            frame_nav_sync = nav.sync_frame_to_track(ts=frame_ts, track=nav_track)
            if frame_nav_sync is not None:
                # sync-derived values fill ONLY missing frame metadata; an
                # operator-supplied frame value always wins (documented
                # precedence: frame > track sync > survey default).  Note:
                # imeta inherits survey defaults as explicit None values, so
                # a plain setdefault would never fire — check for None/absent
                # and write the synced value directly.
                if img.get("latitude") is None and img.get("longitude") is None and imeta.get("lat") is None:
                    imeta["lat"] = frame_nav_sync["latitude"]
                    imeta["lon"] = frame_nav_sync["longitude"]
                    imeta["_nav_synced"] = True
                if imeta.get("heading_deg") is None and frame_nav_sync.get("heading_deg") is not None:
                    imeta["heading_deg"] = frame_nav_sync["heading_deg"]
                    imeta["_nav_heading_synced"] = True
                if imeta.get("altitude_m") is None and frame_nav_sync.get("altitude_m") is not None:
                    imeta["altitude_m"] = frame_nav_sync["altitude_m"]
                    imeta["_nav_altitude_synced"] = True

        detections: list[dict] = []
        for d, did in zip(raw_dets, det_ids, strict=True):
            box = d["box"]
            shadow_info = d["shadow"]
            geo = geolocate.locate_detection(
                meta=imeta,
                slant_range_m=None,
                image_height=processed.shape[0],
                row=box["y"] + box["h"] / 2,
                reference_frame_id=img["id"],
            )
            # v2 provenance enrichment: record sync method, towfish config and
            # normalized status without altering the geometric result.
            if geo.get("known"):
                prov = geo.get("provenance") or {}
                prov["navigation"] = {
                    "sync_method": (frame_nav_sync or {}).get("sync_method"),
                    "nav_source": (frame_nav_sync or {}).get("nav_source")
                    if imeta.get("_nav_synced")
                    else "frame_metadata",
                    "interpolation_fraction": (frame_nav_sync or {}).get("interpolation_fraction"),
                    "gap_s": (frame_nav_sync or {}).get("gap_s"),
                    "towfish_config": {k: v for k, v in tow_cfg.items() if v is not None} or None,
                    "towfish_applied": False,  # abeam engine; offsets recorded for audit
                }
                geo["status"] = "DERIVED"
                geo["location_status"] = "DERIVED"
            else:
                # no target-level fix: classify honestly (§1)
                issues = list(geo.get("provenance", {}).get("issues") or [])
                has_vessel = imeta.get("lat") is not None and imeta.get("lon") is not None
                if has_vessel:
                    # vessel position exists — the missing piece is sonar-side/
                    # geometry (e.g. side not supplied), not navigation.  That
                    # is exactly the VESSEL_ONLY state (§1/§6): never collapse
                    # the vessel fix into the target position.
                    fp = {"latitude": imeta["lat"], "longitude": imeta["lon"], "position_source": "vessel"}
                    geo = {
                        **geolocate.vessel_only(
                            "vessel position available; target-level fix not justifiable from sonar geometry "
                            + (f"({issues[0]})" if issues else ""),
                            fp,
                            geo.get("provenance"),
                        )
                    }
                elif issues and any(
                    "outside valid range" in i or i.startswith("invalid") or "does not exceed" in i for i in issues
                ):
                    geo = {**geolocate.invalid("navigation failed validation", issues, geo.get("provenance"))}
                else:
                    geo = {**geo, "status": "UNAVAILABLE", "location_status": "UNAVAILABLE"}
                if geo.get("status") in ("VESSEL_ONLY", "INVALID", "UNAVAILABLE"):
                    geo.setdefault("location_status", geo["status"])
            dims = dimensions_mod.estimate_dimensions(
                box=box,
                image_height=processed.shape[0],
                range_m=imeta.get("range_m"),
                altitude_m=imeta.get("altitude_m"),
                height_estimate_m=shadow_info.get("height_estimate_m"),
            )
            anom_raw = float(anom_scores.get(did, 0.5))
            phys = d.get("physics") or {}
            physics_score = phys.get("geometry_score") if phys.get("metadata_sufficient") else None
            signals = {
                "detection": float(d["score"]),
                "segmentation": segmentation_evidence(d.get("mask_area_frac")),
                "natural": float(d["p_artificial"]),
                "shadow": d["shadow_evidence"],
                "physics": physics_score,
                "consistency": None,
            }
            availability = {
                "detection": True,
                "segmentation": True,
                "natural": True,
                "shadow": d["shadow_evidence"] is not None,
                "physics": physics_score is not None,
                "consistency": False,
            }
            detections.append(
                {
                    "id": did,
                    "image_id": img["id"],
                    "survey_id": survey_id,
                    "image_filename": img["filename"],
                    "class_name": d["class"],
                    "class_confidence": d["score"],
                    "box": box,
                    "mask_path": mask_paths.get(did),
                    "mask_area_frac": d.get("mask_area_frac"),
                    "evidence": {
                        "signals": signals,
                        "availability": availability,
                        "fusion": 0.0,
                        "breakdown": {},
                        "backend_used": dict(backends),
                    },
                    "status": "candidate",
                    "shadow": shadow_info,
                    "physics": d.get("physics", {}),
                    "geolocation": geo,
                    "dimensions": dims.model_dump(),
                    "priority": {"score": 0.0, "tier": "low", "factors": {}},
                    "raw": {"anomaly_score": anom_raw, "classifier_features": d.get("classifier_features")},
                }
            )
        timings["geolocate"] = (time.perf_counter() - t0) * 1000

        # overlay preview for the Sonar Intelligence Lab
        _write_overlay(processed, detections, settings.storage_dir / f"overlays/{survey_id}/{img['id']}/overlay.png")

        result = {
            "image_id": img["id"],
            "survey_id": survey_id,
            "filename": img["filename"],
            "width": processed.shape[1],
            "height": processed.shape[0],
            "quality": quality.model_dump(),
            "processed_path": proc_rel,
            "preprocess_params": params,
            "detections": detections,
            "backends": backends,
            "backend_notes": warnings,
        }
        db.save_image_result(img["id"], result)
        # persist the navigation-sync provenance on the frame row (§4/§11):
        # which source supplied the fix and how it was time-synchronized.
        if frame_nav_sync is not None:
            with db.transaction() as conn:
                conn.execute(
                    "UPDATE images SET nav_source=?, nav_sync_method=? WHERE id=?",
                    (
                        frame_nav_sync.get("nav_source"),
                        frame_nav_sync.get("sync_method"),
                        img["id"],
                    ),
                )
        # Frame detections are persisted immediately (deterministic stage):
        # target_id is absent here — multi-frame target association is the
        # explicit Phase 2 step, never an implicit side effect.  (The result
        # JSON keeps the pre-fusion per-frame snapshot; the detections table
        # is the canonical, fusion-pass-updated copy the API serves.)
        if detections:
            db.save_detections(detections)
        db.set_image_status(img["id"], "done")
        emit(job_id, "image_done", base + 92, f"Image {i + 1}/{n} complete — {len(detections)} detection(s)")

    # ---------------- survey-level pass ----------------
    if n > 0:
        emit(job_id, "consistency", 96, "Multi-frame consistency verification")
        t0 = time.perf_counter()
        all_dets = db.get_detections(survey_id)
        cons = consistency.compute_consistency(all_dets, distance_m=settings.consistency_distance_m)
        timings["consistency"] = (time.perf_counter() - t0) * 1000

        emit(job_id, "fusion", 98, "Evidence fusion & hazard prioritization")
        t0 = time.perf_counter()
        for d in all_dets:
            c = cons.get(d["id"], {})
            d["evidence"]["signals"]["consistency"] = c.get("evidence")
            d["evidence"]["availability"]["consistency"] = c.get("availability", False)
            d["raw"]["consistency"] = c
            conf, breakdown = fusion.fuse(
                d["evidence"]["signals"],
                d["evidence"]["availability"],
                gain=settings.fusion_gain,
                intercept=settings.fusion_intercept,
            )
            d["evidence"]["fusion"] = conf
            d["evidence"]["breakdown"] = breakdown
            anomaly_raw = d["raw"].get("anomaly_score", 0.0)
            status = fusion.decide_status(
                conf,
                anomaly_raw,
                d["class_confidence"],
                confirm=settings.confirm_threshold,
                review=settings.review_threshold,
            )
            d["status"] = status
            prio = priority_mod.compute_priority(
                class_name=d["class_name"],
                fusion_confidence=conf,
                width_m=d["dimensions"].get("width_m"),
                height_m=d["dimensions"].get("height_m"),
                dimensions_estimable=d["dimensions"].get("estimable", False),
            )
            d["priority"] = prio.model_dump()
        db.update_detections(all_dets)
        timings["fusion"] = (time.perf_counter() - t0) * 1000

        # ---------------- Phase 2: multi-frame association -----------------
        # Frame detections are now grouped into persistent targets with a
        # deterministic, explainable, survey-scoped engine (app.association).
        emit(job_id, "associate", 99, "Multi-frame association — building persistent targets")
        t0 = time.perf_counter()
        from .. import targets as targets_mod

        targets_mod.run_association(survey_id)
        timings["associate"] = (time.perf_counter() - t0) * 1000

    timings["total"] = (time.perf_counter() - t_start) * 1000
    db.update_job(
        job_id,
        status="done",
        stage="done",
        progress=100,
        message=f"Survey analysis complete — {len(db.get_detections(survey_id))} detection(s)",
        timings=timings,
    )
    emit(job_id, "done", 100, "Survey analysis complete")


def _write_overlay(processed: np.ndarray, detections: list[dict], path) -> None:
    """Render the Lab's per-frame overlay PNG (boxes + fused confidence)."""
    vis = processed.copy()
    for d in detections:
        b = d["box"]
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{d['class_name']} {d['evidence']['fusion']:.2f}"
        cv2.putText(vis, label, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)
