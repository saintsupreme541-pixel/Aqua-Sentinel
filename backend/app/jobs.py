"""Job submission, thread-pool execution and SSE event bus."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .pipeline.runner import analyze_survey

_executor: ThreadPoolExecutor | None = None
_event_queues: dict[str, deque[dict]] = {}
_queue_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="aqua-worker")
    return _executor


def emit(job_id: str, stage: str, progress: float, message: str) -> None:
    event = {"stage": stage, "progress": round(progress, 1), "message": message}
    with _queue_lock:
        q = _event_queues.setdefault(job_id, deque(maxlen=1000))
        q.append(event)


def drain_events(job_id: str) -> list[dict]:
    with _queue_lock:
        q = _event_queues.get(job_id)
        if not q:
            return []
        out = list(q)
        q.clear()
    return out


def submit_analysis(survey_id: str, job_id: str) -> None:
    get_executor().submit(_run_guarded, survey_id, job_id)


def _run_guarded(survey_id: str, job_id: str) -> None:
    try:
        analyze_survey(survey_id, job_id, emit)
    except Exception as e:  # pragma: no cover - last-resort guard
        from . import db

        # Keep the database authoritative on failure: job AND its frames are
        # marked failed so persisted state reflects reality after a restart.
        db.update_job(job_id, status="failed", stage="error", error=str(e), message=f"Analysis failed: {e}")
        for img in db.get_images(survey_id):
            if img["status"] == "analyzing":
                db.set_image_status(img["id"], "failed")
        emit(job_id, "failed", 0, f"Analysis failed: {e}")


async def stream_events(job_id: str):
    """Async generator yielding SSE-formatted events until the job finishes.

    Each event is a full Job-shaped payload (status + stage + progress +
    message) so clients can drive their UI from the stream alone; the
    stream ends once the job reaches a terminal state and its queue drains.
    """
    import json

    from . import db

    while True:
        job = db.get_job(job_id)
        status = job["status"] if job else "queued"
        for ev in drain_events(job_id):
            payload = {**ev, "status": status}
            yield f"data: {json.dumps(payload)}\n\n"
        if job and job["status"] in ("done", "failed") and not drain_events(job_id):
            break
        await asyncio.sleep(0.35)
