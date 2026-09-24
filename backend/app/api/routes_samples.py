from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..service import create_survey_with_files

router = APIRouter(prefix="/api/samples", tags=["samples"])


def _samples_dir() -> Path:
    d = settings.sample_data_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("")
def list_samples():
    out = []
    for manifest_path in sorted(_samples_dir().glob("*.json")):
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        image_count = len(m.get("images", []))
        total_kb = 0
        base = _samples_dir() / m.get("images_root", m.get("id", manifest_path.stem))
        for rel in m.get("images", [])[:20]:
            p = base / rel
            if p.exists():
                total_kb += p.stat().st_size
        out.append(
            {
                "id": m.get("id", manifest_path.stem),
                "name": m.get("name", "Sample survey"),
                "description": m.get("description", ""),
                "synthetic": bool(m.get("synthetic", False)),
                "attribution": m.get("attribution", ""),
                "license_note": m.get("license_note", ""),
                "image_count": image_count,
                "size_kb": round(total_kb / 1024.0, 1),
                "has_ground_truth": "ground_truth" in m,
                "tags": m.get("tags", []),
            }
        )
    return out


@router.post("/{sample_id}/import")
def import_sample(sample_id: str):
    manifest_path = _samples_dir() / f"{sample_id}.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"sample '{sample_id}' not found")
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(400, f"invalid manifest: {e}") from e

    base = _samples_dir() / m.get("images_root", sample_id)
    if not base.is_dir():
        raise HTTPException(404, f"sample images for '{sample_id}' not found")
    root = base.resolve()
    entries: list[tuple[str, bytes]] = []
    for rel in m.get("images", []):
        p = (root / rel).resolve()
        if not str(p).startswith(str(root)):
            raise HTTPException(400, f"unsafe path in manifest: {rel}")
        if not p.is_file():
            raise HTTPException(404, f"sample image missing: {rel}")
        entries.append((Path(rel).name, p.read_bytes()))
    if not entries:
        raise HTTPException(400, "manifest lists no images")

    survey_meta = dict(m.get("survey_meta", {}))
    survey_meta.setdefault("name", m.get("name", sample_id))
    survey_meta.setdefault("description", m.get("description", ""))
    if "synthetic" in m:
        survey_meta.setdefault("synthetic", m["synthetic"])
    try:
        sid, job_id = create_survey_with_files(entries, survey_meta, m.get("image_meta") or {})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"survey_id": sid, "job_id": job_id, "image_count": len(entries), "sample_id": sample_id}
