"""SQLite persistence layer.

Single-writer SQLite with a process-global connection guarded by a lock.
All JSON-shaped rows are stored as TEXT and parsed by the API layer.

Conceptual hierarchy (Phase 1 persistence model):

    SURVEY  →  FRAME (the ``images`` table)  →  FRAME DETECTION (``detections``)
                                                     │
                                                     └── target_id ──→  PERSISTENT TARGET

The database is the source of truth: after a server restart every analysis
result, detection and artifact reference is recoverable from these tables.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from .config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "") + "Z"


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS surveys (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            meta_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            image_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'created'
        );
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            survey_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            raw_path TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            frame_index INTEGER,
            captured_at TEXT,
            latitude REAL,
            longitude REAL,
            heading_deg REAL,
            altitude_m REAL,
            sonar_side TEXT,
            slant_range_m REAL,
            FOREIGN KEY (survey_id) REFERENCES surveys(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS image_results (
            image_id TEXT PRIMARY KEY,
            result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            image_id TEXT NOT NULL,
            survey_id TEXT NOT NULL,
            target_id TEXT,
            detection_json TEXT NOT NULL,
            created_at TEXT,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
            FOREIGN KEY (survey_id) REFERENCES surveys(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            survey_id TEXT NOT NULL,
            status TEXT NOT NULL,
            stage TEXT DEFAULT 'queued',
            progress REAL DEFAULT 0,
            message TEXT DEFAULT '',
            error TEXT,
            timings_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            survey_id TEXT NOT NULL,
            format TEXT NOT NULL,
            path TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS targets (
            id TEXT PRIMARY KEY,
            survey_id TEXT NOT NULL,
            canonical_class TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            confidence REAL,
            first_seen_frame_id TEXT,
            last_seen_frame_id TEXT,
            representative_detection_id TEXT,
            latitude REAL,
            longitude REAL,
            geolocation_status TEXT NOT NULL DEFAULT 'unknown',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (survey_id) REFERENCES surveys(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS nav_records (
            id TEXT PRIMARY KEY,
            survey_id TEXT NOT NULL,
            ts TEXT,
            latitude REAL,
            longitude REAL,
            heading_deg REAL,
            altitude_m REAL,
            accuracy_m REAL,
            source TEXT,
            roll REAL,
            pitch REAL,
            heave REAL,
            towfish_x REAL,
            towfish_y REAL,
            towfish_z REAL,
            layback REAL,
            valid INTEGER DEFAULT 1,
            issues_json TEXT DEFAULT '[]',
            created_at TEXT,
            FOREIGN KEY (survey_id) REFERENCES surveys(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_images_survey ON images(survey_id);
        CREATE INDEX IF NOT EXISTS idx_detections_survey ON detections(survey_id);
        CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id);
        CREATE INDEX IF NOT EXISTS idx_targets_survey ON targets(survey_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_survey ON jobs(survey_id);
        CREATE INDEX IF NOT EXISTS idx_reports_survey ON reports(survey_id);
        """
    )
    _add_missing_columns(conn, "images", {"created_at": "TEXT"})
    _add_missing_columns(
        conn,
        "images",
        {
            "frame_index": "INTEGER",
            "captured_at": "TEXT",
            "latitude": "REAL",
            "longitude": "REAL",
            "heading_deg": "REAL",
            "altitude_m": "REAL",
            "sonar_side": "TEXT",
            "slant_range_m": "REAL",
            "nav_source": "TEXT",
            "nav_sync_method": "TEXT",
        },
    )
    _add_missing_columns(conn, "detections", {"target_id": "TEXT", "created_at": "TEXT"})
    # Phase 3: target-level geolocation provenance (source + evidence JSON).
    # Legacy rows keep NULL/defaults — never fabricated retroactively.
    _add_missing_columns(
        conn,
        "targets",
        {"geolocation_source": "TEXT", "geolocation_evidence_json": "TEXT"},
    )
    # Indexes on migrated columns must come AFTER _add_missing_columns — on a
    # pre-Phase-1 database the column only exists from this point on.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_target ON detections(target_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nav_survey_ts ON nav_records(survey_id, ts)")
    conn.commit()


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Idempotent in-place migration: add columns that do not exist yet.

    SQLite's ``ALTER TABLE ADD COLUMN`` cannot run inside ``executescript``'s
    implicit transaction safely across versions, so it is issued per column.
    Existing rows keep NULL for the new columns (navigation metadata is
    nullable by design — never fabricated retroactively).
    """
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}" if prefix else uuid.uuid4().hex


@contextmanager
def transaction():
    """Single-writer transaction scope for multi-row, all-or-nothing writes.

    Yields the connection with an EXCLUSIVE transaction open; commit on clean
    exit, rollback on exception — a failed multi-record write (e.g. a Phase 2
    association run) can never leave half-created relationships behind.
    Nested/implicit commits inside the block are avoided: callers must not
    call ``conn.commit()`` (the context manager owns the boundary).
    """
    with _lock:
        conn = get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# --------------------------------------------------------------------------
# Surveys / images / detections
# --------------------------------------------------------------------------


def create_survey(name: str, description: str, meta: dict[str, Any], image_count: int) -> str:
    sid = new_id("svy_")
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO surveys (id, name, description, meta_json, created_at, image_count) VALUES (?,?,?,?,?,?)",
            (sid, name, description, json.dumps(meta), utcnow(), image_count),
        )
        conn.commit()
    return sid


def add_image(
    sid: str,
    filename: str,
    raw_path: str,
    meta: dict[str, Any],
    width: int,
    height: int,
    *,
    frame_index: int | None = None,
    created_at: str | None = None,
    captured_at: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    heading_deg: float | None = None,
    altitude_m: float | None = None,
    sonar_side: str | None = None,
    slant_range_m: float | None = None,
) -> str:
    iid = new_id("img_")
    with _lock:
        conn = get_conn()
        if frame_index is None:
            frame_index = conn.execute("SELECT COUNT(*) FROM images WHERE survey_id=?", (sid,)).fetchone()[0]
        conn.execute(
            "INSERT INTO images (id, survey_id, filename, raw_path, meta_json, width, height, created_at, "
            "frame_index, captured_at, latitude, longitude, heading_deg, altitude_m, sonar_side, slant_range_m) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                iid,
                sid,
                filename,
                raw_path,
                json.dumps(meta),
                width,
                height,
                created_at or utcnow(),
                frame_index,
                captured_at,
                latitude,
                longitude,
                heading_deg,
                altitude_m,
                sonar_side,
                slant_range_m,
            ),
        )
        conn.commit()
    return iid


def get_survey(sid: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT * FROM surveys WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None


def list_surveys() -> list[dict[str, Any]]:
    with _lock:
        rows = get_conn().execute("SELECT * FROM surveys ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_images(sid: str) -> list[dict[str, Any]]:
    with _lock:
        rows = (
            get_conn()
            .execute("SELECT * FROM images WHERE survey_id=? ORDER BY frame_index, filename", (sid,))
            .fetchall()
        )
    return [dict(r) for r in rows]


def get_image(iid: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    return dict(row) if row else None


def set_image_status(image_id: str, status: str) -> None:
    """Record the analysis lifecycle of a frame in the database itself."""
    with _lock:
        conn = get_conn()
        conn.execute("UPDATE images SET status=? WHERE id=?", (status, image_id))
        conn.commit()


def save_image_result(image_id: str, result: dict[str, Any]) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO image_results (image_id, result_json) VALUES (?,?) "
            "ON CONFLICT(image_id) DO UPDATE SET result_json=excluded.result_json",
            (image_id, json.dumps(result)),
        )
        conn.commit()


def get_image_result(image_id: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT result_json FROM image_results WHERE image_id=?", (image_id,)).fetchone()
    return json.loads(row["result_json"]) if row else None


def save_detections(detections: list[dict[str, Any]]) -> None:
    with _lock:
        conn = get_conn()
        conn.execute("DELETE FROM detections WHERE image_id=?", (detections[0]["image_id"],))
        conn.executemany(
            "INSERT INTO detections (id, image_id, survey_id, target_id, detection_json, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [(d["id"], d["image_id"], d["survey_id"], d.get("target_id"), json.dumps(d), utcnow()) for d in detections],
        )
        conn.commit()


def update_detections(detections: list[dict[str, Any]]) -> None:
    """Upsert detections (used by the survey-level fusion pass).

    ``target_id`` is only written when present in the dict, so a fusion-pass
    update can never clobber an existing association.
    """
    if not detections:
        return
    with _lock:
        conn = get_conn()
        conn.executemany(
            "INSERT INTO detections (id, image_id, survey_id, target_id, detection_json, created_at) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "detection_json=excluded.detection_json, "
            "target_id=COALESCE(excluded.target_id, detections.target_id)",
            [(d["id"], d["image_id"], d["survey_id"], d.get("target_id"), json.dumps(d), utcnow()) for d in detections],
        )
        conn.commit()


def get_detections(survey_id: str | None = None, image_id: str | None = None) -> list[dict[str, Any]]:
    q, args = "SELECT * FROM detections", []
    if survey_id:
        q += " WHERE survey_id=?"
        args.append(survey_id)
    if image_id:
        q += " AND image_id=?" if args else " WHERE image_id=?"
        args.append(image_id)
    q += " ORDER BY detection_json->'$.priority.score' DESC"
    with _lock:
        rows = get_conn().execute(q, args).fetchall()
    out = []
    for r in rows:
        d = json.loads(r["detection_json"])
        d["target_id"] = r["target_id"]  # the column is authoritative
        out.append(d)
    return out


def get_detection_row(detection_id: str) -> dict[str, Any] | None:
    """Fetch one detection row including its scope columns (survey/image/target)."""
    with _lock:
        row = get_conn().execute("SELECT * FROM detections WHERE id=?", (detection_id,)).fetchone()
    if not row:
        return None
    d = json.loads(row["detection_json"])
    d["target_id"] = row["target_id"]
    return d


# --------------------------------------------------------------------------
# Persistent targets (Phase 1 foundation — creation stays conservative and
# deterministic; multi-frame association belongs to Phase 2)
# --------------------------------------------------------------------------


def create_target(
    survey_id: str,
    canonical_class: str,
    *,
    status: str = "active",
    confidence: float | None = None,
    first_seen_frame_id: str | None = None,
    last_seen_frame_id: str | None = None,
    representative_detection_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    geolocation_status: str = "unknown",
    notes: str = "",
) -> str:
    tid = new_id("tgt_")
    now = utcnow()
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO targets (id, survey_id, canonical_class, status, confidence, first_seen_frame_id, "
            "last_seen_frame_id, representative_detection_id, latitude, longitude, geolocation_status, notes, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                tid,
                survey_id,
                canonical_class,
                status,
                confidence,
                first_seen_frame_id,
                last_seen_frame_id,
                representative_detection_id,
                latitude,
                longitude,
                geolocation_status,
                notes,
                now,
                now,
            ),
        )
        conn.commit()
    return tid


def get_target(tid: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT * FROM targets WHERE id=?", (tid,)).fetchone()
    return dict(row) if row else None


def get_targets(survey_id: str) -> list[dict[str, Any]]:
    with _lock:
        rows = (
            get_conn().execute("SELECT * FROM targets WHERE survey_id=? ORDER BY created_at", (survey_id,)).fetchall()
        )
    return [dict(r) for r in rows]


def update_target(tid: str, **fields: Any) -> None:
    """Update whitelisted target columns; always refreshes ``updated_at``."""
    allowed = {
        "status",
        "confidence",
        "first_seen_frame_id",
        "last_seen_frame_id",
        "representative_detection_id",
        "latitude",
        "longitude",
        "geolocation_status",
        "geolocation_source",
        "geolocation_evidence_json",
        "notes",
    }
    sets, args = [], []
    for k, v in fields.items():
        if k in allowed and v is not None:
            sets.append(f"{k}=?")
            args.append(v)
    if not sets:
        return
    sets.append("updated_at=?")
    args.extend([utcnow(), tid])
    with _lock:
        conn = get_conn()
        conn.execute(f"UPDATE targets SET {', '.join(sets)} WHERE id=?", args)
        conn.commit()


def delete_target(tid: str) -> bool:
    with _lock:
        conn = get_conn()
        cur = conn.execute("DELETE FROM targets WHERE id=?", (tid,))
        conn.commit()
    return cur.rowcount > 0


def assign_detection_target(detection_id: str, target_id: str | None) -> None:
    """Link (or unlink) one frame detection to a persistent target.

    The target must belong to the same survey as the detection — enforced
    here so a detection can never leak across surveys.
    """
    with _lock:
        conn = get_conn()
        det = conn.execute("SELECT survey_id FROM detections WHERE id=?", (detection_id,)).fetchone()
        if not det:
            raise ValueError(f"detection {detection_id} not found")
        if target_id is not None:
            tgt = conn.execute("SELECT survey_id FROM targets WHERE id=?", (target_id,)).fetchone()
            if not tgt:
                raise ValueError(f"target {target_id} not found")
            if tgt["survey_id"] != det["survey_id"]:
                raise ValueError("target belongs to a different survey")
        conn.execute("UPDATE detections SET target_id=? WHERE id=?", (target_id, detection_id))
        conn.commit()


# --------------------------------------------------------------------------
# Navigation records (geolocation v2 — genuine uploaded navigation only)
# --------------------------------------------------------------------------


def replace_nav_records(survey_id: str, records: list[dict[str, Any]]) -> int:
    """Replace a survey's navigation track with a validated set.

    Only records that PASSED validation (``valid=1``) are ever inserted;
    rejected rows are reported by the caller's diagnostics, never stored as
    usable navigation.  One transaction: a failed upload cannot leave a
    half-imported track behind.
    """
    with _lock:
        conn = get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM nav_records WHERE survey_id=?", (survey_id,))
            conn.executemany(
                "INSERT INTO nav_records (id, survey_id, ts, latitude, longitude, heading_deg, altitude_m, "
                "accuracy_m, source, roll, pitch, heave, towfish_x, towfish_y, towfish_z, layback, "
                "valid, issues_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        r.get("id") or new_id("nav_"),
                        survey_id,
                        r.get("ts"),
                        r.get("latitude"),
                        r.get("longitude"),
                        r.get("heading_deg"),
                        r.get("altitude_m"),
                        r.get("accuracy_m"),
                        r.get("source"),
                        r.get("roll"),
                        r.get("pitch"),
                        r.get("heave"),
                        r.get("towfish_x"),
                        r.get("towfish_y"),
                        r.get("towfish_z"),
                        r.get("layback"),
                        1 if r.get("valid", True) else 0,
                        json.dumps(r.get("issues", [])),
                        utcnow(),
                    )
                    for r in records
                ],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return len(records)


def get_nav_records(survey_id: str, *, valid_only: bool = True) -> list[dict[str, Any]]:
    q = "SELECT * FROM nav_records WHERE survey_id=?"
    if valid_only:
        q += " AND valid=1"
    q += " ORDER BY ts"
    with _lock:
        rows = get_conn().execute(q, (survey_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["valid"] = bool(d["valid"])
        d["issues"] = json.loads(d.get("issues_json") or "[]")
        out.append(d)
    return out


def nav_stats(survey_id: str) -> dict[str, int]:
    with _lock:
        conn = get_conn()
        total = conn.execute("SELECT COUNT(*) FROM nav_records WHERE survey_id=?", (survey_id,)).fetchone()[0]
        valid = conn.execute("SELECT COUNT(*) FROM nav_records WHERE survey_id=? AND valid=1", (survey_id,)).fetchone()[
            0
        ]
    return {"total": total, "valid": valid, "rejected": total - valid}


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


def create_job(survey_id: str) -> str:
    jid = new_id("job_")
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO jobs (id, survey_id, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (jid, survey_id, "queued", utcnow(), utcnow()),
        )
        conn.commit()
    return jid


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
    timings: dict[str, float] | None = None,
) -> None:
    fields: list[str] = []
    args: list[Any] = []
    if status is not None:
        fields.append("status=?")
        args.append(status)
    if stage is not None:
        fields.append("stage=?")
        args.append(stage)
    if progress is not None:
        fields.append("progress=?")
        args.append(progress)
    if message is not None:
        fields.append("message=?")
        args.append(message)
    if error is not None:
        fields.append("error=?")
        args.append(error)
    if timings is not None:
        fields.append("timings_json=?")
        args.append(json.dumps(timings))
    if not fields:
        return
    fields.append("updated_at=?")
    args.append(utcnow())
    args.append(job_id)
    with _lock:
        conn = get_conn()
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id=?", args)
        conn.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["timings"] = json.loads(d.get("timings_json") or "{}")
    return d


def latest_job(survey_id: str) -> dict[str, Any] | None:
    with _lock:
        row = (
            get_conn()
            .execute("SELECT * FROM jobs WHERE survey_id=? ORDER BY created_at DESC LIMIT 1", (survey_id,))
            .fetchone()
        )
    if not row:
        return None
    d = dict(row)
    d["timings"] = json.loads(d.get("timings_json") or "{}")
    return d


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def create_report(survey_id: str, fmt: str, path: str, size_bytes: int) -> str:
    rid = new_id("rep_")
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO reports (id, survey_id, format, path, size_bytes, created_at) VALUES (?,?,?,?,?,?)",
            (rid, survey_id, fmt, path, size_bytes, utcnow()),
        )
        conn.commit()
    return rid


def list_reports(survey_id: str) -> list[dict[str, Any]]:
    with _lock:
        rows = (
            get_conn()
            .execute("SELECT * FROM reports WHERE survey_id=? ORDER BY created_at DESC", (survey_id,))
            .fetchall()
        )
    return [dict(r) for r in rows]


def get_report(rid: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
    return dict(row) if row else None
