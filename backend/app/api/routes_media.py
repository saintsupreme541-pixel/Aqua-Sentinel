from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..storage import safe_relative

router = APIRouter(prefix="/api/media", tags=["media"])

_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
}


@router.get("/{path:path}")
def serve_media(path: str):
    try:
        rel = safe_relative(path)
    except ValueError:
        raise HTTPException(400, "invalid path") from None
    p = settings.storage_dir / rel
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(p, media_type=_TYPES.get(p.suffix.lower(), "application/octet-stream"))
