"""Unit tests for OCR visual telemetry extraction module."""

import cv2
import numpy as np
from backend.app.pipeline.ocr_telemetry import extract_visual_telemetry_from_bytes, parse_ddm_or_dms


def test_parse_ddm_or_dms():
    # Test Merged DDM Lat/Lon
    lat = parse_ddm_or_dms("Latitude 32477817 N")
    assert lat is not None
    assert abs(lat - 32.7963617) < 0.0001

    lon = parse_ddm_or_dms("Longtud@ 034555366 E")
    assert lon is not None
    assert abs(lon - 34.92561) < 0.0001

    # Test Separated DDM
    lat2 = parse_ddm_or_dms("32:47.7817 N")
    assert lat2 is not None
    assert abs(lat2 - 32.7963617) < 0.0001

    # Test Decimal Degrees
    lat3 = parse_ddm_or_dms("32.7963617 N")
    assert lat3 is not None
    assert abs(lat3 - 32.7963617) < 0.0001


def test_extract_visual_telemetry_from_blank_image():
    # Empty / blank image returns empty dict without crashing
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", blank)
    res = extract_visual_telemetry_from_bytes(buf.tobytes())
    assert isinstance(res, dict)
