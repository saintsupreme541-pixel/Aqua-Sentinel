"""Multi-frame / multi-ping consistency verification.

Canonical owner: :mod:`app.association` (Phase 2).  This module is a thin
re-export so existing imports keep working; the geospatial grouping + the
image-space fallback live in ONE implementation consumed by both the survey
runner and the single-image analysis service — no divergent algorithms.
"""

from __future__ import annotations

from ..association import compute_consistency

__all__ = ["compute_consistency"]
