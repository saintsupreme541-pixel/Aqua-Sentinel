from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import db
from .api import (
    routes_analyze,
    routes_dashboard,
    routes_detections,
    routes_frames,
    routes_geolocation,
    routes_jobs,
    routes_media,
    routes_reports,
    routes_samples,
    routes_surveys,
)
from .config import settings
from .models.registry import load_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("aqua")

# Single-port demo mode: if the built frontend exists, serve it under /app.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    db.get_conn()
    log.info("AQUA-SENTINEL backend ready")
    log.info("storage dir: %s", settings.storage_dir)
    log.info("model registry: %s", settings.model_registry_path)
    if _FRONTEND_DIST.is_dir():
        log.info("frontend build found — serving dashboard at /app")
    yield


app = FastAPI(
    title="AQUA-SENTINEL AI",
    description=(
        "Sonar-aware marine debris intelligence pipeline — detection, verification, geolocation, prioritization."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_surveys.router)
app.include_router(routes_frames.router)
app.include_router(routes_samples.router)
app.include_router(routes_jobs.router)
app.include_router(routes_detections.router)
app.include_router(routes_reports.router)
app.include_router(routes_media.router)
app.include_router(routes_dashboard.router)
app.include_router(routes_analyze.router)
app.include_router(routes_geolocation.router)

if _FRONTEND_DIST.is_dir():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


@app.get("/api/health")
def health():
    reg = load_registry()
    backends = {task: cfg.get("backend", "heuristic") for task, cfg in reg.items()}
    return {
        "status": "ok",
        "service": "aqua-sentinel",
        "version": "0.1.0",
        "model_backends": backends,
        "mode": "full-model" if any(v != "heuristic" for v in backends.values()) else "degraded-heuristic",
    }
