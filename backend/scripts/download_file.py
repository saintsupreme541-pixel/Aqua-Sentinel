"""Resilient dataset downloader.

Handles the figshare ``202 Accepted`` (file preparation) dance, follows
redirect chains, and streams to disk with progress logging.  Usage::

    python download_file.py <url> <output>

Exit code 0 only when the output file exists and is non-empty.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
CHUNK = 1 << 20


def download(url: str, out: Path, max_prep_wait: int = 300) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    headers = {"User-Agent": UA}

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=120.0), headers=headers) as client:
        # Phase 1: wait out 202 "preparing" responses (figshare large files).
        waited = 0
        while True:
            with client.stream("GET", url) as r:
                if r.status_code == 202:
                    if waited >= max_prep_wait:
                        print(f"ERROR: still 202 after {waited}s", flush=True)
                        return False
                    print(f"202 preparing... waited {waited}s", flush=True)
                    time.sleep(15)
                    waited += 15
                    continue
                if r.status_code != 200:
                    print(f"ERROR: HTTP {r.status_code}", flush=True)
                    return False
                total = int(r.headers.get("content-length", 0))
                print(f"downloading {url} -> {out} ({total / 1e9:.2f} GB)", flush=True)
                done = 0
                t0 = time.time()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if done % (100 * CHUNK) < CHUNK:
                            rate = done / max(time.time() - t0, 1e-6) / 1e6
                            print(f"  {done / 1e9:.2f}/{total / 1e9:.2f} GB ({rate:.1f} MB/s)", flush=True)
                break

    if tmp.stat().st_size == 0:
        print("ERROR: empty file", flush=True)
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(out)
    print(f"DONE: {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)
    return True


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        raise SystemExit(2)
    url, out = sys.argv[1], Path(sys.argv[2])
    wait = int(sys.argv[3]) if len(sys.argv) == 4 else 300
    ok = download(url, out, max_prep_wait=wait)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
