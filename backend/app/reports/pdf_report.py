"""Professional PDF report via ReportLab."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..config import settings

ACCENT = colors.HexColor("#0891b2")
DARK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
TIER_COLORS = {
    "critical": colors.HexColor("#dc2626"),
    "high": colors.HexColor("#f59e0b"),
    "medium": colors.HexColor("#eab308"),
    "low": colors.HexColor("#16a34a"),
}


def render(data: dict, out_path: str | Path) -> None:
    survey = data["survey"]
    meta = data["survey_meta"]
    dets = data["detections"]

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"AQUA-SENTINEL — {survey['name']}",
        author="AQUA-SENTINEL AI",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=DARK, fontSize=20, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=MUTED, fontSize=9, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=ACCENT, fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)

    story: list = []
    story.append(Paragraph("AQUA-SENTINEL AI", h1))
    story.append(Paragraph("Marine Intelligence Report — from sonar data to ocean action", sub))
    story.append(Paragraph(f"Survey: {survey['name']}", h2))
    story.append(Paragraph(f"ID: {survey['id']} &nbsp;·&nbsp; Created: {survey['created_at']}", body))
    if meta.get("description"):
        story.append(Paragraph(f"Description: {meta['description']}", body))

    meta_rows = [
        ["Sonar type", str(meta.get("sonar_type", "unknown"))],
        ["GPS", "available" if meta.get("lat") is not None else "not recorded"],
        ["Heading", str(meta.get("heading_deg", "—")) + "°"],
        ["Altitude", f"{meta['altitude_m']} m" if meta.get("altitude_m") is not None else "—"],
        ["Range", f"{meta['range_m']} m" if meta.get("range_m") is not None else "—"],
        ["Preprocessing preset", str(meta.get("preprocess_preset", "light"))],
        ["Images analyzed", str(len(data["images"]))],
        ["Detections", str(len(dets))],
    ]
    story.append(_meta_table(meta_rows))

    story.append(Paragraph("Detections", h2))
    if not dets:
        story.append(Paragraph("No debris detections in this survey.", body))
    for i, d in enumerate(dets, 1):
        story.extend(_detection_block(data, d, i, styles))

    doc.build(story)


def _style_table(rows, w1, w2):
    t = Table(rows, colWidths=[w1, w2])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return t


def _meta_table(rows):
    return _style_table(rows, 45 * mm, 90 * mm)


def _detection_block(data: dict, d: dict, i: int, styles) -> list:
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.5, leading=12)
    ev = d.get("evidence", {})
    prio = d.get("priority", {})
    geo = d.get("geolocation", {})
    dims = d.get("dimensions", {})
    shadow = d.get("shadow", {})

    tier_color = TIER_COLORS.get(prio.get("tier", "low"), MUTED)
    h2tier = ParagraphStyle(
        "h2tier", parent=styles["Heading2"], textColor=tier_color, fontSize=12, spaceBefore=8, spaceAfter=2
    )
    out: list = []
    out.append(Paragraph(f"{i}. {d.get('class_name', 'debris').upper()} — {prio.get('tier', 'low').upper()}", h2tier))
    box = d.get("box", {})
    box_txt = f"{box.get('x')}, {box.get('y')}, {box.get('w')}, {box.get('h')}"
    dims_txt = f"{dims.get('width_m')} m across-track" if dims.get("estimable") else (dims.get("note") or "—")
    h_txt = f"{shadow.get('height_estimate_m')} m" if shadow.get("height_estimate_m") else "not estimated"
    rows = [
        ["Status", d.get("status", "")],
        ["Fusion confidence", f"{ev.get('fusion', 0):.2f}"],
        ["Priority score", f"{prio.get('score', 0):.1f} / 100"],
        ["Image", d.get("image_filename", "")],
        ["Box (x, y, w, h)", box_txt],
        [
            "Location",
            (
                f"≈ {geo.get('lat'):.5f}, {geo.get('lon'):.5f} (approximate, ±{geo.get('uncertainty_m')} m)"
                if geo.get("known")
                else "Unavailable — insufficient navigation/sonar metadata"
            ),
        ],
        [
            "Location source",
            ("Frame navigation + sonar geometry (derived, approximate)" if geo.get("known") else "—"),
        ],
        ["Dimensions", dims_txt],
        ["Height (shadow)", h_txt],
    ]
    out.append(_style_table(rows, 45 * mm, 90 * mm))

    ev_rows = [["Evidence signal", "Value", "Available"]]
    for key, label in [
        ("detection", "AI detection"),
        ("segmentation", "Segmentation agreement"),
        ("natural", "Artificial probability"),
        ("shadow", "Acoustic shadow"),
        ("consistency", "Multi-frame consistency"),
        ("anomaly", "Known-distribution fit"),
    ]:
        sig = (ev.get("signals") or {}).get(key)
        avail = (ev.get("availability") or {}).get(key, False)
        ev_rows.append([label, f"{sig:.2f}" if sig is not None else "—", "yes" if avail else "no"])
    ev_table = Table(ev_rows, colWidths=[60 * mm, 40 * mm, 35 * mm])
    ev_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ]
        )
    )
    out.append(Spacer(1, 3))
    out.append(ev_table)

    mask_path = d.get("mask_path")
    if mask_path:
        p = settings.storage_dir / mask_path
        if p.exists():
            out.append(Spacer(1, 4))
            out.append(Image(str(p), width=70 * mm, height=70 * mm * p_aspect(p)))

    out.append(Spacer(1, 2))
    note = (
        "<i>Uncertainty: location is estimated from survey metadata + sonar geometry; never survey-grade. "
        "Frame position ≠ object position — the object offset is derived from sonar geometry "
        "(flat-seabed assumption). Missing metadata means location is reported as unavailable, "
        f"never fabricated. Backends: {ev.get('backend_used', {})}.</i>"
    )
    out.append(Paragraph(note, body))
    out.append(Spacer(1, 4))
    return out


def p_aspect(p: Path) -> float:
    from PIL import Image as PILImage

    with PILImage.open(p) as im:
        w, h = im.size
    return min(1.0, h / max(w, 1))
