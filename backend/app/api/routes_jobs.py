from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import db, jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
def job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@router.get("/{job_id}/stream")
async def job_stream(job_id: str):
    if not db.get_job(job_id):
        raise HTTPException(404, "job not found")
    return StreamingResponse(
        jobs.stream_events(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
