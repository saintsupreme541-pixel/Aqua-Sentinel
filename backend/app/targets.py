"""Survey-level multi-frame association orchestrator (Phase 2).

Turns frame detections into persistent targets:

    frames (frame_index ascending) → detections → association → targets

Properties:

- **Survey-scoped** — every query and write is restricted to one survey;
  cross-survey association is impossible (also enforced in ``db``).
- **Idempotent** — running twice yields the same targets.  Links that are
  still valid are reused ("reused" decisions); unassociated detections are
  associated deterministically; ``reset=True`` explicitly clears all links
  and targets before re-running (for re-analyzed surveys).  Nothing is
  silently duplicated.
- **Transactional** — all writes of one run happen in a single SQLite
  transaction; a failure leaves no half-created relationships.
- **Explainable** — every decision is stored on the detection
  (``raw.association``) and aggregated in the run summary.

Deterministic rules (documented, not invented at call sites):

- Processing order: ``frame_index`` ascending; stable fallback
  (created_at, id) when missing.  Never random DB order.
- Representative detection: highest evidence-fusion score; ties break on
  lowest (frame_index, image id, detection id) — stable across runs.
- first/last_seen: min/max associated frame by the same order — actual
  observations, never target-creation time.
- Target confidence: max fusion of associated detections (a documented
  prototype aggregate, NOT a calibrated probability).
- Target location: representative detection's geolocation when known,
  else the linked detection with the lowest uncertainty; status
  ``approximate`` (sonar-geometry estimate) or ``unknown`` — never
  fabricated.  Phase 3: the chosen rule keeps the representative *target*
  location aligned with the representative *detection* (its fix wins when
  it is located); the fallback differs only when the representative
  detection has no fix, and the provenance records exactly which
  observation supplied the coordinate and why.
- Image-space (spatial) signal: only when every frame in the survey shares
  the same processed dimensions (comparable coordinate space).
"""

from __future__ import annotations

import json
import logging

from . import db
from .association import DEFAULTS, rank_candidates
from .pipeline import geolocate

log = logging.getLogger("aqua.targets")


def _ordered_frames(images: list[dict]) -> list[dict]:
    """Deterministic processing order (§21): frame_index ascending, stable
    (created_at, id) fallback when the index is missing."""
    return sorted(
        images,
        key=lambda im: (
            im["frame_index"] if im.get("frame_index") is not None else 10**9,
            im.get("created_at") or "",
            im["id"],
        ),
    )


def run_association(survey_id: str, *, reset: bool = False) -> dict:
    """Associate all detections of one survey into persistent targets.

    Deterministic, survey-scoped, idempotent, transactional (see module
    docstring).  Returns the run summary (frames/detections processed,
    targets created, associated/ambiguous/unassociated counts).
    """
    frames = _ordered_frames(db.get_images(survey_id))
    detections = db.get_detections(survey_id)
    if not frames:
        return {
            "survey_id": survey_id,
            "frames_processed": 0,
            "detections_processed": 0,
            "targets_created": 0,
            "detections_associated": 0,
            "already_associated": 0,
            "unassociated": 0,
            "ambiguous": 0,
            "target_ids": [],
        }

    frame_index_of = {im["id"]: im.get("frame_index") for im in frames}
    dims_of = {im["id"]: (im["width"], im["height"]) for im in frames}
    # spatial signal is only meaningful in a single shared coordinate space
    uniform_dims = len({dims_of[fid] for fid in dims_of}) == 1

    det_by_frame: dict[str, list[dict]] = {}
    for d in detections:
        det_by_frame.setdefault(d["image_id"], []).append(d)

    created_targets: list[str] = []
    decisions: dict[str, dict] = {}
    n_assoc = n_already = n_new_target = n_unassoc = n_ambig = 0

    with db.transaction() as conn:
        # ---- explicit, safe reset (§23): only when asked -------------------
        if reset:
            conn.execute("UPDATE detections SET target_id=NULL WHERE survey_id=?", (survey_id,))
            conn.execute("DELETE FROM targets WHERE survey_id=?", (survey_id,))

        # existing links (idempotency: reused, never re-created).  NOTE: after
        # a reset these must reflect the just-cleared DB, not the stale
        # pre-transaction snapshot — so they are always re-read here.
        existing_links = {
            r["id"]: r["target_id"]
            for r in conn.execute(
                "SELECT id, target_id FROM detections WHERE survey_id=? AND target_id IS NOT NULL",
                (survey_id,),
            ).fetchall()
        }
        targets_by_id = {
            r["id"]: dict(r) for r in conn.execute("SELECT * FROM targets WHERE survey_id=?", (survey_id,)).fetchall()
        }

        # in-memory candidate state per target (mutable as the sweep proceeds)
        linked_by_target: dict[str, list[dict]] = {tid: [] for tid in targets_by_id}
        for d in detections:
            tid = existing_links.get(d["id"])
            if tid:
                linked_by_target.setdefault(tid, []).append({**d, "_frame_index": frame_index_of.get(d["image_id"])})

        def candidate_state(tid: str) -> dict:
            """Live target view for the engine (class + geo + rep box + last frame)."""
            trow = targets_by_id[tid]
            linked = linked_by_target.get(tid, [])
            rep = next((d for d in linked if d["id"] == trow["representative_detection_id"]), None)
            if rep is None and linked:
                rep = sorted(linked, key=lambda d: d["id"])[0]
            last_idx = max((d["_frame_index"] for d in linked if d["_frame_index"] is not None), default=None)
            return {
                "id": tid,
                "canonical_class": trow["canonical_class"],
                "latitude": trow.get("latitude"),
                "longitude": trow.get("longitude"),
                "representative_box": rep.get("box") if rep else None,
                "last_frame_index": last_idx,
            }

        for frame in frames:
            fid = frame["id"]
            fidx = frame.get("frame_index")
            for det in sorted(det_by_frame.get(fid, []), key=lambda d: d["id"]):
                det_view = {**det, "frame_index": fidx}

                # (a) already linked → reuse (idempotent re-run, §22)
                if det["id"] in existing_links:
                    n_already += 1
                    decisions[det["id"]] = {
                        "associated": True,
                        "status": "reused",
                        "target_id": existing_links[det["id"]],
                        "score": None,
                        "signals": {},
                        "unavailable": [],
                        "reasons": ["existing association reused (idempotent re-run)"],
                    }
                    continue

                # (b) score against the survey's current targets
                if targets_by_id:
                    result = rank_candidates(
                        det_view,
                        [candidate_state(tid) for tid in sorted(targets_by_id)],
                        same_dimensions=uniform_dims,
                    )
                else:
                    result = {
                        "associated": False,
                        "status": "no_candidates",
                        "score": 0.0,
                        "candidates": [],
                        "unavailable": [],
                        "reasons": ["no existing targets yet — seeding a new target"],
                    }

                if result.get("associated"):
                    tid = result["target_id"]
                    known = bool(det.get("geolocation", {}).get("known"))
                    targets_by_id[tid].setdefault("latitude", None)
                    if known:
                        targets_by_id[tid]["latitude"] = det["geolocation"].get("lat")
                        targets_by_id[tid]["longitude"] = det["geolocation"].get("lon")
                    linked_by_target.setdefault(tid, []).append({**det, "_frame_index": fidx})
                    n_assoc += 1
                    decisions[det["id"]] = result
                elif result.get("status") == "ambiguous":
                    # §19: two viable targets within the margin → no arbitrary
                    # merge; the detection stays unassociated, flagged for review.
                    n_ambig += 1
                    decisions[det["id"]] = result
                else:
                    # (c) below threshold / no candidates → new target.
                    # Conservative by design: a wrong merge destroys
                    # information; an extra target just awaits an operator.
                    tid = db.new_id("tgt_")
                    now = db.utcnow()
                    conn.execute(
                        "INSERT INTO targets (id, survey_id, canonical_class, status, notes, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (
                            tid,
                            survey_id,
                            det["class_name"],
                            "active",
                            f"auto-created from detection {det['id']} (association run: {result.get('status')})",
                            now,
                            now,
                        ),
                    )
                    targets_by_id[tid] = {
                        "id": tid,
                        "survey_id": survey_id,
                        "canonical_class": det["class_name"],
                        "representative_detection_id": None,
                        "latitude": det["geolocation"].get("lat") if det["geolocation"].get("known") else None,
                        "longitude": det["geolocation"].get("lon") if det["geolocation"].get("known") else None,
                        "confidence": None,
                        "first_seen_frame_id": None,
                        "last_seen_frame_id": None,
                        "geolocation_status": "unknown",
                        "notes": "",
                        "status": "active",
                    }
                    linked_by_target[tid] = [{**det, "_frame_index": fidx}]
                    created_targets.append(tid)
                    n_new_target += 1
                    decisions[det["id"]] = {
                        **result,
                        "associated": True,
                        "status": "new_target",
                        "target_id": tid,
                        "reasons": [*result.get("reasons", []), "no acceptable existing target — new target created"],
                    }

        # ---- apply links + persist decision explanations -------------------
        for det in detections:
            decision = decisions.get(det["id"])
            if not decision:
                continue
            tid = decision.get("target_id")
            if tid and det["id"] not in existing_links:
                conn.execute(
                    "UPDATE detections SET target_id=? WHERE id=? AND survey_id=?",
                    (tid, det["id"], survey_id),
                )
            raw = det.get("raw") or {}
            raw["association"] = decision
            conn.execute(
                "UPDATE detections SET detection_json=? WHERE id=?",
                (json.dumps({**det, "raw": raw}), det["id"]),
            )

        # ---- recompute derived target state (same transaction) -------------
        for tid in targets_by_id:
            _recompute_target_state_conn(conn, tid, survey_id)
        # Phase 3: idempotent — recomputation is deterministic over the
        # linked detections, so a repeated run yields identical state.

    summary = {
        "survey_id": survey_id,
        "frames_processed": len(frames),
        "detections_processed": len(detections),
        "targets_created": len(created_targets),
        "detections_associated": n_assoc,
        "already_associated": n_already,
        "unassociated": n_unassoc,
        "ambiguous": n_ambig,
        "target_ids": sorted(targets_by_id.keys()),
        "config": {
            "class_weight": DEFAULTS.class_weight,
            "geo_weight": DEFAULTS.geo_weight,
            "frame_weight": DEFAULTS.frame_weight,
            "spatial_weight": DEFAULTS.spatial_weight,
            "accept_threshold": DEFAULTS.accept_threshold,
            "ambiguity_margin": DEFAULTS.ambiguity_margin,
            "geo_full_score_m": DEFAULTS.geo_full_score_m,
            "geo_zero_score_m": DEFAULTS.geo_zero_score_m,
        },
    }
    log.info(
        "association survey=%s frames=%d dets=%d targets=%d(+%d) assoc=%d reused=%d ambig=%d",
        survey_id,
        len(frames),
        len(detections),
        len(targets_by_id),
        len(created_targets),
        n_assoc,
        n_already,
        n_ambig,
    )
    return summary


def _recompute_target_state_conn(conn, tid: str, survey_id: str) -> None:
    """Recompute derived target columns from its linked detections.

    Runs inside the caller's transaction; rules documented in the module
    docstring (representative / first / last / confidence / location).
    """
    now = db.utcnow()
    linked = [
        {**json.loads(r["detection_json"]), "target_id": r["target_id"]}
        for r in conn.execute(
            "SELECT detection_json, target_id FROM detections WHERE survey_id=?", (survey_id,)
        ).fetchall()
        if r["target_id"] == tid
    ]
    if not linked:
        return
    frames = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM images WHERE survey_id=?", (survey_id,)).fetchall()}

    def frame_key(d: dict) -> tuple:
        im = frames.get(d["image_id"], {})
        fi = im.get("frame_index") if im else None
        return (fi if fi is not None else 10**9, d["image_id"])

    ordered = sorted(linked, key=frame_key)

    def rep_key(d: dict) -> tuple:
        return (-(d.get("evidence", {}).get("fusion") or 0.0), *frame_key(d), d["id"])

    rep = min(linked, key=rep_key)
    located = [d for d in linked if d.get("geolocation", {}).get("known") and d["geolocation"].get("lat") is not None]
    lat = lon = None
    geo_status = "unknown"
    geo_source: str | None = None
    geo_evidence: dict = {"kind": "aggregate", "observations_located": len(located), "n_observations": len(linked)}
    if located:
        # §8: prefer the representative *detection*'s fix so the target
        # location and the representative observation never disagree; fall
        # back to the lowest-uncertainty located observation only when the
        # representative has no fix (recorded in the provenance).
        rep_located = next((d for d in located if d["id"] == rep["id"]), None)
        if rep_located is not None:
            pick, pick_rule = rep_located, "representative detection (lowest fusion tiebreak: phase-2 rule)"
        else:
            pick = min(located, key=lambda d: (d["geolocation"].get("uncertainty_m") or 10**9, d["id"]))
            pick_rule = "lowest-uncertainty located observation (representative detection has no fix)"
        lat, lon = pick["geolocation"]["lat"], pick["geolocation"]["lon"]
        geo_status = "approximate"
        geo_source = "frame_navigation_plus_sonar_geometry"
        prov = pick["geolocation"].get("provenance") or {}
        geo_evidence = {
            **geo_evidence,
            "kind": "aggregate",
            "status": "approximate",
            "source": geo_source,
            "selection_rule": pick_rule,
            "from_detection_id": pick["id"],
            "from_frame_id": pick["image_id"],
            "derived": prov.get("derived", {}),
            "assumptions": prov.get("assumptions", []),
            "note": (
                "target coordinate is the representative observation's "
                "approximate fix — never averaged, never fabricated"
            ),
        }

    # Multi-frame location consistency (§17/§18): distinct from association.
    lc = geolocate.location_consistency(
        [(d["geolocation"]["lat"], d["geolocation"]["lon"]) for d in located],
        [d["geolocation"].get("uncertainty_m") for d in located],
    )
    geo_evidence["location_consistency"] = lc

    conn.execute(
        "UPDATE targets SET first_seen_frame_id=?, last_seen_frame_id=?, representative_detection_id=?, "
        "confidence=?, latitude=?, longitude=?, geolocation_status=?, geolocation_source=?, "
        "geolocation_evidence_json=?, updated_at=? WHERE id=?",
        (
            ordered[0]["image_id"],
            ordered[-1]["image_id"],
            rep["id"],
            max((d.get("evidence", {}).get("fusion") or 0.0) for d in linked),
            lat,
            lon,
            geo_status,
            geo_source,
            json.dumps(geo_evidence),
            now,
            tid,
        ),
    )


def target_history(survey_id: str, tid: str) -> dict:
    """Full target detail with its detection history in deterministic frame order."""
    t = db.get_target(tid)
    if not t or t["survey_id"] != survey_id:
        return {}
    frames = {im["id"]: im for im in db.get_images(survey_id)}
    linked = [d for d in db.get_detections(survey_id) if d.get("target_id") == tid]

    def frame_key(d: dict) -> tuple:
        im = frames.get(d["image_id"], {})
        fi = im.get("frame_index") if im else None
        return (fi if fi is not None else 10**9, d["image_id"], d["id"])

    history = []
    for d in sorted(linked, key=frame_key):
        im = frames.get(d["image_id"], {})
        history.append(
            {
                "detection_id": d["id"],
                "frame_id": d["image_id"],
                "frame_index": im.get("frame_index"),
                "frame_filename": im.get("filename"),
                "class_name": d["class_name"],
                "class_confidence": d.get("class_confidence"),
                "fusion": d.get("evidence", {}).get("fusion"),
                "status": d.get("status"),
                "box": d.get("box"),
                "mask_path": d.get("mask_path"),
                "geolocation": d.get("geolocation"),
                "association": (d.get("raw") or {}).get("association"),
            }
        )
    rep = next((h for h in history if h["detection_id"] == t.get("representative_detection_id")), None)
    evidence = json.loads(t.get("geolocation_evidence_json") or "null")
    lc = (evidence or {}).get("location_consistency") or {}
    return {
        "id": t["id"],
        "survey_id": t["survey_id"],
        "canonical_class": t["canonical_class"],
        "status": t["status"],
        "confidence": t.get("confidence"),
        "confidence_definition": (
            "max evidence-fusion score of associated detections (prototype aggregate, not a calibrated probability)"
        ),
        "first_seen_frame_id": t.get("first_seen_frame_id"),
        "last_seen_frame_id": t.get("last_seen_frame_id"),
        "representative_detection_id": t.get("representative_detection_id"),
        "representative": rep,
        "latitude": t.get("latitude"),
        "longitude": t.get("longitude"),
        "geolocation_status": t.get("geolocation_status"),
        "geolocation_source": t.get("geolocation_source"),
        "geolocation_evidence": evidence,
        "location_consistency": lc
        or {"available": False, "status": "insufficient_data", "dispersion_m": None, "score": None},
        "location_note": "approximate: estimated from frame navigation + sonar geometry (never survey-grade)"
        if t.get("geolocation_status") == "approximate"
        else "location unavailable — no observation had enough valid navigation/sonar metadata to derive one",
        "notes": t.get("notes", ""),
        "created_at": t.get("created_at"),
        "updated_at": t.get("updated_at"),
        "n_observations": len(history),
        "history": history,
    }
