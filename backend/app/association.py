"""Multi-frame association: decide which frame detections are the SAME object.

The core question (Phase 2): when two frames both contain a "debris"
detection, do those boxes show one physical target seen twice — or two
different objects?  Four pictures of the same car are one car, not four.

Design rules:

- **Deterministic** — same inputs always produce the same associations.
  No RNG, no learned model, no tracker networks (explicitly out of scope).
- **Explainable** — every accept/reject carries a structured, inspectable
  evidence record (per-signal scores + human-readable reasons).
- **Conservative** — a wrong merge destroys information; a missed merge
  merely leaves two targets for an operator to merge later.  Ambiguous
  candidates are never merged arbitrarily.
- **Honest about missing data** — unavailable evidence is excluded and
  weights renormalised (never scored as zero); AQUA is vehicle-agnostic,
  so GPS/heading/altitude/side may all be absent.

This module owns BOTH the detection→target association scoring AND the
multi-frame consistency *evidence* reporting (see §Consistency below) so
runner.py and analysis_service.py cannot drift apart again.

All thresholds/weights are documented prototype heuristics (docs/association.md),
not scientifically validated universal values.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline.geolocate import haversine_distance

# ---------------------------------------------------------------------------
# Configuration — named, documented prototype heuristics
# ---------------------------------------------------------------------------


@dataclass
class AssociationConfig:
    """Prototype association parameters (documented, not validated)."""

    class_weight: float = 0.35
    geo_weight: float = 0.30
    frame_weight: float = 0.15
    spatial_weight: float = 0.20

    # class compatibility (§7): same class = strong positive; different class
    # = strong negative but NOT an absolute ban (a detector can genuinely flip
    # a borderline object; the negative weight simply makes it near-impossible
    # to clear the accept threshold without overwhelming other evidence).
    class_same_score: float = 1.0
    class_diff_score: float = 0.0
    class_conflict_penalty: float = 0.35  # multiplied into the final score

    # Conflicting *positional* evidence is a strong negative too: when both
    # observations ARE located but demonstrably far apart (> geo_zero_score_m),
    # or when comparable adjacent frames show disjoint boxes (IoU ≈ 0), the
    # renormalised mean must not be allowed to outvote the conflict.
    geo_conflict_penalty: float = 0.35
    spatial_conflict_penalty: float = 0.50

    # frame sequence proximity (§8): score decays with frame_index gap
    frame_full_score_gap: float = 1.0  # gap ≤ this → full score
    frame_zero_score_gap: float = 8.0  # gap ≥ this → zero score

    # geographic consistency (§9): score decays with distance
    geo_full_score_m: float = 8.0  # ≤ this distance → full score
    geo_zero_score_m: float = 40.0  # ≥ this distance → zero score

    # image-space consistency (§11): only used between comparable frames
    # (same dimensions).  Distance = IoU distance between rep boxes.
    spatial_full_iou: float = 0.55  # IoU ≥ this → full score
    spatial_zero_iou: float = 0.05  # IoU ≤ this → zero score

    accept_threshold: float = 0.62
    ambiguity_margin: float = 0.06  # top-two within this margin → ambiguous


DEFAULTS = AssociationConfig()


# ---------------------------------------------------------------------------
# Per-signal scoring primitives
# ---------------------------------------------------------------------------


def _linear_decay(value: float, full_at: float, zero_at: float) -> float:
    """1.0 when value ≤ full_at, linearly → 0.0 at zero_at, clamped to [0,1]."""
    if value <= full_at:
        return 1.0
    if value >= zero_at:
        return 0.0
    return 1.0 - (value - full_at) / (zero_at - full_at)


def class_signal(det_class: str, target_class: str) -> tuple[float, list[str]]:
    if det_class == target_class:
        return DEFAULTS.class_same_score, ["same class"]
    return DEFAULTS.class_diff_score, []


def frame_proximity_signal(frame_idx_a: int | None, frame_idx_b: int | None) -> tuple[float | None, dict, list[str]]:
    """Frame-sequence proximity from ``frame_index`` (never treated as time)."""
    if frame_idx_a is None or frame_idx_b is None:
        return None, {"frame_gap": None}, ["frame order unknown — proximity unavailable"]
    gap = abs(frame_idx_a - frame_idx_b)
    score: float = _linear_decay(float(gap), DEFAULTS.frame_full_score_gap, DEFAULTS.frame_zero_score_gap)
    return score, {"frame_gap": gap}, [f"frames {gap} apart"]


def geo_signal(geo_a: dict | None, geo_b: dict | None) -> tuple[float | None, dict, list[str]]:
    """Geographic consistency via haversine distance — only when BOTH are known."""
    if not (geo_a and geo_a.get("known") and geo_a.get("lat") is not None):
        return None, {"geographic_distance_m": None}, ["geolocation unavailable"]
    if not (geo_b and geo_b.get("known") and geo_b.get("lat") is not None):
        return None, {"geographic_distance_m": None}, ["target geolocation unavailable"]
    dist = haversine_distance(geo_a["lat"], geo_a["lon"], geo_b["lat"], geo_b["lon"])
    score = _linear_decay(dist, DEFAULTS.geo_full_score_m, DEFAULTS.geo_zero_score_m)
    return score, {"geographic_distance_m": round(dist, 1)}, [f"{dist:.1f} m apart"]


def _iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def spatial_signal(box_a: dict, box_b: dict) -> tuple[float, dict, list[str]]:
    """Image-space consistency: IoU of the raw bounding boxes.

    Only computed between frames that share the same processed-image
    dimensions (the caller guarantees comparability — see targets.py);
    raw pixel coordinates across differently-scaled images are meaningless.
    """
    iou = _iou(box_a, box_b)
    score = _linear_decay(1.0 - iou, 1.0 - DEFAULTS.spatial_full_iou, 1.0 - DEFAULTS.spatial_zero_iou)
    return score, {"box_iou": round(iou, 3)}, [f"box IoU {iou:.2f}"]


# ---------------------------------------------------------------------------
# Detection→target association
# ---------------------------------------------------------------------------


def score_detection_vs_target(det: dict, target: dict, *, same_dimensions: bool = True) -> dict:
    """Score one frame detection against one existing target.

    Returns the structured evidence record (§18): per-signal availability +
    scores, renormalised weighted mean over AVAILABLE signals only, reasons,
    unavailable list, and the final accepted/ambiguous verdict.
    """
    reasons: list[str] = []
    unavailable: list[str] = []

    cls_score, cls_reasons = class_signal(det["class_name"], target["canonical_class"])
    reasons.extend(cls_reasons)

    signals: dict[str, dict] = {
        "class": {"available": True, "score": round(cls_score, 3)},
    }
    weighted: list[tuple[float, float]] = [(DEFAULTS.class_weight, cls_score)]

    # --- geographic consistency (when both sides are genuinely located) ----
    t_geo = {"known": target.get("latitude") is not None, "lat": target.get("latitude"), "lon": target.get("longitude")}
    geo_score, geo_details, geo_reasons = geo_signal(det.get("geolocation"), t_geo)
    if geo_score is None:
        unavailable.append("geographic")
        # If EITHER side lacks geolocation the signal is unavailable — the
        # reason strings from geo_signal already say which side.
        reasons.extend(geo_reasons[:1])
    else:
        signals["geographic"] = {"available": True, **geo_details, "score": round(geo_score, 3)}
        weighted.append((DEFAULTS.geo_weight, geo_score))
        reasons.extend(geo_reasons)

    # --- frame sequence proximity ------------------------------------------
    fp_raw, fp_details, fp_reasons = frame_proximity_signal(det.get("frame_index"), target.get("last_frame_index"))
    if fp_raw is None:
        unavailable.append("frame_proximity")
    else:
        fp_score: float = fp_raw  # narrowed for the type checker
        signals["frame_proximity"] = {"available": True, **fp_details, "score": round(fp_score, 3)}
        weighted.append((DEFAULTS.frame_weight, fp_score))
        reasons.extend(fp_reasons)

    # --- image-space consistency (comparable frames only) ------------------
    if same_dimensions and target.get("representative_box") and det.get("box"):
        sp_score, sp_details, sp_reasons = spatial_signal(det["box"], target["representative_box"])
        signals["spatial"] = {"available": True, **sp_details, "score": round(sp_score, 3)}
        weighted.append((DEFAULTS.spatial_weight, sp_score))
        reasons.extend(sp_reasons)
    else:
        unavailable.append("spatial")

    # --- weighted renormalisation over AVAILABLE signals -------------------
    total_w = sum(w for w, _ in weighted)
    score = sum(w * s for w, s in weighted) / total_w if total_w > 0 else 0.0

    # Conflicts are applied AFTER renormalisation so a conflicting match can
    # only win by overwhelming the conflicting evidence (which the penalties
    # make practically impossible — see §7/§20: prefer conservative splits).
    conflict = det["class_name"] != target["canonical_class"]
    if conflict:
        score *= DEFAULTS.class_conflict_penalty
        reasons.append("class mismatch penalty applied")

    geo_sig = signals.get("geographic")
    if (
        geo_sig is not None
        and geo_sig.get("geographic_distance_m") is not None
        and geo_sig["geographic_distance_m"] > DEFAULTS.geo_zero_score_m
    ):
        score *= DEFAULTS.geo_conflict_penalty
        reasons.append(
            f"geographic conflict: {geo_sig['geographic_distance_m']} m apart exceeds "
            f"the {DEFAULTS.geo_zero_score_m:.0f} m association window"
        )

    sp_sig = signals.get("spatial")
    fp_sig = signals.get("frame_proximity")
    if (
        sp_sig is not None
        and sp_sig.get("box_iou") is not None
        and sp_sig["box_iou"] <= 0.0
        and fp_sig is not None
        and fp_sig.get("frame_gap") is not None
        and fp_sig["frame_gap"] <= DEFAULTS.frame_full_score_gap
    ):
        # comparable adjacent frames with disjoint boxes: weak-but-real
        # negative evidence — not a veto, but enough to stay conservative
        score *= DEFAULTS.spatial_conflict_penalty
        reasons.append("disjoint boxes in comparable adjacent frames (spatial conflict penalty)")

    accepted = score >= DEFAULTS.accept_threshold
    return {
        "associated": accepted,
        "target_id": target["id"],
        "score": round(score, 4),
        "signals": signals,
        "unavailable": unavailable,
        "reasons": reasons,
        "class_conflict": conflict,
    }


def rank_candidates(det: dict, targets: list[dict], *, same_dimensions: bool = True) -> dict:
    """Rank all candidate targets for one detection and resolve ambiguity (§19).

    Wrong merges are worse than unmatched detections: when the top two
    candidates are within ``ambiguity_margin`` the result is explicitly
    ``ambiguous`` and the detection stays unassociated.
    """
    scored = [score_detection_vs_target(det, t, same_dimensions=same_dimensions) for t in targets]
    scored.sort(key=lambda e: e["score"], reverse=True)
    if not scored:
        return {
            "associated": False,
            "status": "no_candidates",
            "score": 0.0,
            "candidates": [],
            "unavailable": [],
            "reasons": ["no existing targets in survey"],
        }
    best = scored[0]
    ambiguous = len(scored) > 1 and (best["score"] - scored[1]["score"]) < DEFAULTS.ambiguity_margin
    if best["score"] < DEFAULTS.accept_threshold:
        return {
            "associated": False,
            "status": "below_threshold",
            "score": best["score"],
            "best_candidate": best,
            "candidates": scored[:5],
            "unavailable": best.get("unavailable", []),
            "reasons": [f"best score {best['score']:.2f} < accept threshold {DEFAULTS.accept_threshold}"],
        }
    if ambiguous:
        return {
            "associated": False,
            "status": "ambiguous",
            "score": best["score"],
            "best_candidate": best,
            "candidates": scored[:5],
            "unavailable": best.get("unavailable", []),
            "reasons": [
                (
                    f"top two candidates within ambiguity margin "
                    f"({best['score']:.2f} vs {scored[1]['score']:.2f}) — not merged"
                )
            ],
        }
    return {"associated": True, "status": "accepted", **best, "candidates": scored[:5]}


# ---------------------------------------------------------------------------
# §28 — one canonical multi-frame consistency implementation
# ---------------------------------------------------------------------------


def compute_consistency(
    detections: list[dict], *, distance_m: float = 25.0, per_image_dims: dict[str, tuple[int, int]] | None = None
) -> dict[str, dict]:
    """Consistency evidence for fusion, derived from the association signals.

    Grouping rule (documented heuristic): detections in *different* frames
    whose estimated geolocations are within ``distance_m`` are treated as
    views of the same physical object and reinforce each other.  When
    geolocation is unavailable for the pair, image-space IoU between
    same-dimension frames is used as a weaker fallback signal.

    Single-image surveys (or surveys without geolocation) report
    availability=False — never fabricated temporal evidence.
    """
    by_image: dict[str, list[dict]] = {}
    for d in detections:
        by_image.setdefault(d["image_id"], []).append(d)
    multi_image = len(by_image) >= 2

    out: dict[str, dict] = {}
    for d in detections:
        geo = d.get("geolocation", {})
        known = bool(geo.get("known") and geo.get("lat") is not None)
        n_geo = 0
        n_spatial = 0
        for other in detections:
            if other["id"] == d["id"] or other["image_id"] == d["image_id"]:
                continue
            og = other.get("geolocation", {})
            if known and og.get("known") and og.get("lat") is not None:
                if haversine_distance(geo["lat"], geo["lon"], og["lat"], og["lon"]) <= distance_m:
                    n_geo += 1
            elif (
                per_image_dims
                and per_image_dims.get(d["image_id"]) == per_image_dims.get(other["image_id"])
                and d.get("box")
                and other.get("box")
            ):
                sp, _, _ = spatial_signal(d["box"], other["box"])
                if sp is not None and sp >= DEFAULTS.spatial_full_iou:
                    n_spatial += 1

        if not multi_image:
            out[d["id"]] = {
                "evidence": None,
                "availability": False,
                "n_views": 1 if known else 0,
                "note": "single image — multi-frame consistency not assessable",
            }
            continue

        n_views = 1 + n_geo + n_spatial
        if n_geo + n_spatial > 0:
            out[d["id"]] = {
                "evidence": round(min(1.0, (n_geo + n_spatial) / 2.0), 3),
                "availability": True,
                "n_views": n_views,
                "note": f"seen in {n_views} survey images",
            }
        else:
            out[d["id"]] = {
                "evidence": 0.0,
                "availability": True,
                "n_views": 1,
                "note": "no corroborating view in other frames",
            }
    return out
