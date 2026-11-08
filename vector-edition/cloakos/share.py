"""Share training results with the CloakOS project on Supabase, so runs on your Mac and Claude's
cloud runs land in one place to compare and train from.

What goes up: a run's settings, its progress curve, its exam score, and the trained policy files.
Calibration results too. Never camera images, audio, or anything about you.

    python -m cloakos.share run data/policies/goto-map-20261027-120000 --source cloud
    python -m cloakos.share calibration data/calibration.json

Turn it off with share.enabled: false in config.yaml, or CLOAK_SHARE=0.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import uuid
from pathlib import Path

import httpx

from . import NOUS_VERSION, __version__

log = logging.getLogger(__name__)

URL = "https://ufhzuntiekxtsipkhmwn.supabase.co"
KEY = "sb_publishable_OUqDktcbAMWJN7Y8Wt3-CA_NdtSeuwu"          # publishable: only lets the app add rows


def enabled() -> bool:
    return os.environ.get("CLOAK_SHARE", "1") != "0"


def machine() -> str:
    return f"{platform.system()} {platform.machine()} · {os.cpu_count()} cores"


def _post(table: str, row: dict, timeout: float = 60.0) -> None:
    r = httpx.post(f"{URL}/rest/v1/{table}", json=row, timeout=timeout,
                   headers={"apikey": KEY, "Authorization": f"Bearer {KEY}", "Prefer": "return=minimal"})
    if r.status_code >= 300:
        raise RuntimeError(f"{table}: {r.status_code} {r.text[:200]}")


def upload_run(folder: str | Path, source: str = "local") -> str:
    """Send one finished training run. Returns its id."""
    d = Path(folder)
    meta = json.loads((d / "meta.json").read_text())
    rows = [json.loads(x) for x in (d / "progress.jsonl").read_text().splitlines() if x.strip()] if (d / "progress.jsonl").exists() else []
    step = max(1, len(rows) // 1000)
    exam = json.loads((d / "exam.json").read_text()) if (d / "exam.json").exists() else None
    last = rows[-1] if rows else {}
    run_id = str(uuid.uuid4())
    _post("training_runs", {
        "id": run_id, "source": source, "machine": machine(), "app_version": __version__, "nous_version": NOUS_VERSION,
        "run_name": d.name, "task": meta.get("task"), "sensors": meta.get("sensors"),
        "steps": meta.get("trained_steps") or last.get("steps"), "wall_s": last.get("t"), "fps": last.get("fps"),
        "meta": meta, "progress": rows[::step], "final": last, "exam": exam,
    })
    for name in ("policy.npz", "model.zip"):
        p = d / name
        if p.exists() and p.stat().st_size < 5_000_000:
            _post("training_policies", {"run_id": run_id, "name": name, "data": "\\x" + p.read_bytes().hex()}, timeout=180.0)
    return run_id


def upload_calibration(result: dict, target: str | None, source: str = "local") -> None:
    _post("calibrations", {"source": source, "machine": machine(), "app_version": __version__, "target": target, "result": result})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cloakos.share")
    sub = ap.add_subparsers(dest="what", required=True)
    r = sub.add_parser("run"); r.add_argument("folder"); r.add_argument("--source", default="local", choices=("local", "cloud"))
    c = sub.add_parser("calibration"); c.add_argument("file"); c.add_argument("--source", default="local", choices=("local", "cloud"))
    a = ap.parse_args(argv)
    if a.what == "run":
        print("uploaded run", upload_run(a.folder, a.source))
    else:
        res = json.loads(Path(a.file).read_text())
        upload_calibration(res, res.get("target"), a.source)
        print("uploaded calibration")


if __name__ == "__main__":
    main()
