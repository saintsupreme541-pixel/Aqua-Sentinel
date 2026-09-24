"""Pydantic schemas shared across the API, job runner and reports.

These are the source of truth for the JSON contract consumed by the
frontend. Keep field names stable — the frontend types mirror them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SurveyMeta(BaseModel):
    """Survey-level metadata (applies to all images unless overridden)."""

    name: str = "Untitled Survey"
    description: str = ""
    sonar_type: Literal["sss", "fls", "other", "unknown"] = "unknown"
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    altitude_m: float | None = None
    range_m: float | None = None
    side: Literal["port", "starboard", "unknown"] = "unknown"
    track_points: list[dict[str, float]] = Field(default_factory=list)
    preprocess_preset: str = "light"


class ImageMeta(SurveyMeta):
    """Per-image metadata; overrides survey defaults."""

    filename: str = ""
    captured_at: str | None = None


class QualityReport(BaseModel):
    score: float = Field(ge=0, le=100)
    flags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata_completeness: dict[str, bool] = Field(default_factory=dict)


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class EvidenceSignals(BaseModel):
    detection: float | None = None
    segmentation: float | None = None
    natural: float | None = None  # probability that object is artificial
    shadow: float | None = None
    physics: float | None = None  # sonar-geometry plausibility score
    consistency: float | None = None
    anomaly: float | None = None  # inverted: 1 - anomaly_score


class EvidenceResult(BaseModel):
    signals: EvidenceSignals = Field(default_factory=EvidenceSignals)
    availability: dict[str, bool] = Field(default_factory=dict)
    fusion: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    backend_used: dict[str, str] = Field(default_factory=dict)


class PhysicsInfo(BaseModel):
    """Physics-informed quantities derived from shadow + sonar geometry.

    ``None`` fields mean *not estimable* — never fabricated.
    """

    metadata_sufficient: bool = False
    grazing_angle_deg: float | None = None
    estimated_height_m: float | None = None
    geometry_score: float | None = None
    note: str = ""


class ShadowInfo(BaseModel):
    available: bool = False
    valid: bool | None = None
    length_px: float | None = None
    length_m: float | None = None
    height_estimate_m: float | None = None
    note: str = ""


class Geolocation(BaseModel):
    """Detection geolocation with Phase 3 provenance.

    ``status``: ``approximate`` (derived position) or ``unknown`` (missing/
    invalid metadata — never zero, never fabricated).  ``provenance``
    separates OBSERVED inputs (frame navigation) from DERIVED quantities
    (bearing, ground range, position) and records assumptions + issues.
    """

    known: bool = False
    status: Literal["approximate", "unknown"] = "unknown"
    lat: float | None = None
    lon: float | None = None
    uncertainty_m: float | None = None
    ellipse: dict[str, float] | None = None  # semi-axes + rotation for display
    provenance: dict[str, Any] | None = None
    note: str = ""


class Dimensions(BaseModel):
    estimable: bool = False
    width_m: float | None = None  # across-track
    length_m: float | None = None  # along-track
    height_m: float | None = None  # from shadow geometry
    note: str = ""


class PriorityFactors(BaseModel):
    type_risk: float = 0.0
    size: float = 0.0
    entanglement: float = 0.0
    environmental: float = 0.0
    confidence: float = 0.0
    location: float = 0.0


class PriorityResult(BaseModel):
    score: float = Field(default=0.0, ge=0, le=100)
    tier: Literal["critical", "high", "medium", "low"] = "low"
    factors: PriorityFactors = Field(default_factory=PriorityFactors)


class Detection(BaseModel):
    id: str
    image_id: str
    survey_id: str
    image_filename: str = ""
    class_name: str = "debris"
    class_confidence: float = 0.0  # raw detector score
    box: BBox
    mask_path: str | None = None
    mask_area_frac: float | None = None
    evidence: EvidenceResult = Field(default_factory=EvidenceResult)
    status: Literal["confirmed", "review", "candidate", "human_review_required"] = "candidate"
    shadow: ShadowInfo = Field(default_factory=ShadowInfo)
    physics: PhysicsInfo = Field(default_factory=PhysicsInfo)
    geolocation: Geolocation = Field(default_factory=Geolocation)
    dimensions: Dimensions = Field(default_factory=Dimensions)
    priority: PriorityResult = Field(default_factory=PriorityResult)
    raw: dict[str, Any] = Field(default_factory=dict)


class ImageResult(BaseModel):
    image_id: str
    survey_id: str
    filename: str
    width: int
    height: int
    quality: QualityReport | None = None
    processed_path: str | None = None
    preprocess_params: dict[str, Any] = Field(default_factory=dict)
    detections: list[Detection] = Field(default_factory=list)
    backends: dict[str, str] = Field(default_factory=dict)
    backend_notes: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    job_id: str
    survey_id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ReportInfo(BaseModel):
    id: str
    survey_id: str
    format: Literal["csv", "json", "geojson", "pdf"]
    path: str
    size_bytes: int
    created_at: datetime = Field(default_factory=_utcnow)
