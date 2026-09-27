"""Optical Character Recognition (OCR) Visual Telemetry Extractor.

Extracts on-screen display (OSD) telemetry metadata (Latitude, Longitude, Heading,
Pitch, Roll, Altitude, Slant Range) printed visually on sonar software screenshots
(e.g. Klein, SonarWiz, EdgeTech status bars).
"""

from __future__ import annotations

import asyncio
import io
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_ddm_or_dms(text_val: str) -> float | None:
    """Parse DDM (32 47 7817 N / 32:47.7817 N) or DD (32.79636 N) to decimal degrees."""
    if not text_val:
        return None

    # Pattern 1: Merged 8/9-digit DDM: 32477817 N or 034555366 E
    m_merged_lat = re.search(r'(\d{2})(\d{2})(\d{4})\s*([NS])', text_val, re.I)
    if m_merged_lat:
        deg, min_w, min_f, ref = m_merged_lat.groups()
        val = float(deg) + float(f"{min_w}.{min_f}") / 60.0
        return round(-val if ref.upper() == "S" else val, 7)

    m_merged_lon = re.search(r'0?(\d{2})(\d{2})(\d{4})\s*([EW])', text_val, re.I)
    if m_merged_lon:
        deg, min_w, min_f, ref = m_merged_lon.groups()
        val = float(deg) + float(f"{min_w}.{min_f}") / 60.0
        return round(-val if ref.upper() == "W" else val, 7)

    # Pattern 2: Explicit separated DDM: 32:47.7817 N or 32 47.7817 N or 32 deg 47.7817 N
    m_ddm = re.search(r'(\d{1,3})[\s°:]+(\d{1,2}(?:\.\d+)?)\s*([NSEW])', text_val, re.I)
    if m_ddm:
        deg, minutes, ref = m_ddm.groups()
        val = float(deg) + float(minutes) / 60.0
        return round(-val if ref.upper() in ("S", "W") else val, 7)

    # Pattern 3: Standard decimal degrees: 32.796361 N
    m_dd = re.search(r'([-+]?\d{1,3}\.\d+)\s*([NSEW])?', text_val, re.I)
    if m_dd:
        val = float(m_dd.group(1))
        ref = m_dd.group(2)
        if ref and ref.upper() in ("S", "W"):
            val = -val
        return round(val, 7)

    return None


def extract_visual_telemetry_from_bytes(content: bytes) -> dict[str, Any]:
    """Extract visual telemetry text from sonar acquisition software screenshot bytes."""
    extracted: dict[str, Any] = {}
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return extracted

    h, w, _ = img.shape
    # Crop bottom status bar (bottom 25% of screen)
    status_bar = img[int(h * 0.75) : h, 0:w]
    gray = cv2.cvtColor(cv2.resize(status_bar, (0, 0), fx=3, fy=3), cv2.COLOR_BGR2GRAY)

    temp_path = Path("temp_ocr_frame.png").resolve()
    cv2.imwrite(str(temp_path), gray)

    lines: list[str] = []
    try:
        from winsdk.windows.graphics.imaging import BitmapDecoder
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage import StorageFile

        async def _run_win_ocr():
            file = await StorageFile.get_file_from_path_async(str(temp_path))
            stream = await file.open_async(0)
            decoder = await BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            engine = OcrEngine.try_create_from_user_profile_languages()
            if not engine:
                return []
            res = await engine.recognize_async(bitmap)
            return [line.text for line in res.lines]

        lines = asyncio.run(_run_win_ocr())
    except Exception:
        pass
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

    if not lines:
        return extracted

    full_text = " ".join(lines)

    # Latitude
    for line in lines:
        if any(c in line for c in ("N", "S", "Latitude", "Lat")):
            val = parse_ddm_or_dms(line)
            if val is not None and -90.0 <= val <= 90.0:
                extracted["lat"] = val
                break

    # Longitude
    for line in lines:
        if any(c in line for c in ("E", "W", "Longitude", "Long", "Lon")):
            val = parse_ddm_or_dms(line)
            if val is not None and -180.0 <= val <= 180.0:
                extracted["lon"] = val
                break

    # Heading / Course
    hdg_m = re.search(r"(?:heading|course)[\s:]*(\d{1,3}(?:\.\d+)?)", full_text, re.I)
    if hdg_m:
        val = float(hdg_m.group(1))
        if 0.0 <= val <= 360.0:
            extracted["heading_deg"] = round(val, 1)

    # Pitch
    pitch_m = re.search(r"pitch[\s:]*([-+]?\d+(?:\.\d+)?)", full_text, re.I)
    if pitch_m:
        extracted["pitch_deg"] = float(pitch_m.group(1))

    # Roll
    roll_m = re.search(r"roll[\s:]*([-+]?\d+(?:\.\d+)?)", full_text, re.I)
    if roll_m:
        extracted["roll_deg"] = float(roll_m.group(1))

    # Altitude
    alt_m = re.search(r"alt(?:itude)?[\s:]*(\d+(?:\.\d+)?)", full_text, re.I)
    if alt_m:
        extracted["altitude_m"] = float(alt_m.group(1))

    # Range
    rng_m = re.search(r"(?:s\s*range|range)[\s:]*(\d+(?:\.\d+)?)", full_text, re.I)
    if rng_m:
        extracted["range_m"] = float(rng_m.group(1))

    return extracted
