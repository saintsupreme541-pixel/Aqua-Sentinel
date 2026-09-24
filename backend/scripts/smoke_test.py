"""Live-server smoke test for AQUA-SENTINEL.

Boots uvicorn on a spare port against a *fresh* temp storage dir, then
exercises the API end to end over real HTTP:

    health → samples list → sample import (SSE stream observed) → survey
    detail → detections → per-image lab result → reports (CSV/JSON/GeoJSON/
    PDF downloads) → media artifact → static frontend mount (/app when a
    frontend/dist build exists).

Usage::

    python scripts/smoke_test.py [--port 8123]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent


def wait_ready(base: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/api/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit("backend did not become ready in time")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    venv = ROOT / "backend" / ".venv"
    python = (venv / "Scripts" / "python.exe") if os.name == "nt" else venv / "bin" / "python"

    with tempfile.TemporaryDirectory(prefix="aqua-smoke-") as td:
        env = {**os.environ, "AQUA_STORAGE_DIR": td, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            [str(python), "-m", "uvicorn", "app.main:app", "--port", str(args.port), "--log-level", "warning"],
            cwd=str(ROOT / "backend"),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            wait_ready(base)
            checks: list[str] = []

            def ok(name: str, cond: bool, extra: str = "") -> None:
                checks.append(name)
                status = "PASS" if cond else "FAIL"
                print(f"  [{status}] {name} {extra}")
                if not cond:
                    sys.exit(1)

            print("smoke test against live server…")
            ok("health", httpx.get(f"{base}/api/health").json().get("status") == "ok")

            samples = httpx.get(f"{base}/api/samples").json()
            ids = {s["id"] for s in samples}
            ok(
                "samples listed (synthetic + watertank)",
                {"synthetic_sss", "md_fls_watertank", "md_fls_watertank_demo"} <= ids,
            )

            imp = httpx.post(f"{base}/api/samples/synthetic_sss/import").json()
            sid, job_id = imp["survey_id"], imp["job_id"]
            ok("sample import returns survey+job", bool(sid and job_id), f"({imp['image_count']} images)")

            # SSE stream: every event must carry the full Job-shaped status field
            saw_sse = False
            try:
                with httpx.stream("GET", f"{base}/api/jobs/{job_id}/stream", timeout=15.0) as resp:
                    if "text/event-stream" in resp.headers.get("content-type", ""):
                        for line in resp.iter_lines():
                            if line.startswith("data: "):
                                ev = json.loads(line[6:])
                                if ev.get("status") in ("done", "failed") or ev.get("stage") == "done":
                                    saw_sse = True
                                    if ev.get("status") not in ("done", "failed"):
                                        ok("SSE event carries status", "status" in ev, str(list(ev)))
                                    break
            except Exception:
                pass
            ok("SSE event stream", saw_sse)

            job = None
            for _ in range(300):
                job = httpx.get(f"{base}/api/jobs/{job_id}").json()
                if job["status"] in ("done", "failed"):
                    break
                time.sleep(0.5)
            ok(
                "job completes",
                job is not None and job["status"] == "done",
                f"({job['timings'].get('total', 0):.0f} ms total)",
            )

            detail = httpx.get(f"{base}/api/surveys/{sid}").json()
            ok("survey detail lists images", detail["image_count"] == 7)

            imgs = detail["images"]
            ok("detail images carry raw_url", all("raw_url" in im for im in imgs))
            dets = httpx.get(f"{base}/api/surveys/{sid}/detections").json()
            ok("detections present", len(dets) > 0, f"({len(dets)} detections)")

            im_res = httpx.get(f"{base}/api/surveys/{sid}/images/{imgs[0]['id']}").json()
            ok("lab image result w/ urls", bool(im_res["raw_url"] and im_res.get("processed_url")), im_res["raw_url"])
            ok(
                "image result exposes backends + notes",
                "backend_notes" in im_res and "anomaly" in im_res.get("backends", {}),
            )
            ov = httpx.get(base + im_res["overlay_url"])
            ok("overlay artifact served", ov.status_code == 200 and ov.headers["content-type"] == "image/png")
            raw = httpx.get(base + im_res["raw_url"])
            ok("raw artifact served", raw.status_code == 200)

            gj = httpx.get(f"{base}/api/surveys/{sid}/geojson").json()
            ok("geojson export valid", gj["type"] == "FeatureCollection" and len(gj["features"]) >= 0)

            rep = httpx.post(f"{base}/api/surveys/{sid}/reports", json=["csv", "json", "geojson", "pdf"])
            ok("reports generated (4 formats)", rep.status_code == 200 and len(rep.json()["report_ids"]) == 4)
            for rid in rep.json()["report_ids"]:
                dl = httpx.get(f"{base}/api/reports/{rid}/download")
                ok(
                    f"report download {rid[:12]}",
                    dl.status_code == 200 and len(dl.content) > 0,
                    f"({len(dl.content)} bytes)",
                )
            pdf_id = rep.json()["report_ids"][-1]
            pdf = httpx.get(f"{base}/api/reports/{pdf_id}/download").content
            ok("pdf is a real PDF", pdf[:4] == b"%PDF", f"header {pdf[:4]!r}")

            dash = httpx.get(f"{base}/api/dashboard").json()
            ok("dashboard aggregates", dash["survey_count"] >= 1 and dash["detection_count"] >= 0)

            app_resp = httpx.get(f"{base}/app", follow_redirects=True)
            if app_resp.status_code == 200 and "aqua" in app_resp.text.lower():
                ok("frontend mounted at /app", True)
                # Regression guard: index.html must reference assets under /app/
                # (Vite base="/app/"), not the origin root — otherwise the
                # single-port demo serves a blank page.
                srcs = re.findall(r'(?:src|href)="([^"]+)"', app_resp.text)
                assets = [s for s in srcs if "/assets/" in s]
                if assets:
                    ok(
                        "frontend assets under /app base",
                        all(a.startswith("/app/assets/") for a in assets),
                        f"refs: {assets}",
                    )
                    asset = httpx.get(f"{base}{assets[0]}")
                    ok("frontend asset resolves", asset.status_code == 200, f"({asset.status_code})")
                else:
                    ok("frontend asset refs found in index.html", False, app_resp.text[:120])
            else:
                print("  [SKIP] frontend /app mount (dist build not present)")

            print(f"\nall {len(checks)} smoke checks passed ✔")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
