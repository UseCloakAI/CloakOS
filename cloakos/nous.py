"""Nous model updates: fetch the newest shipped policies without reinstalling the app.

The packaged policies live in cloakos/rl/pretrained with a manifest.json. Newer ones are
downloaded into <home>/nous and loaded in preference to the packaged copy.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import httpx

from . import __version__
from .rl.skills import PRETRAINED

log = logging.getLogger(__name__)
REPO = "UseCloakAI/CloakOS"


def override_dir() -> Path:
    h = os.environ.get("CLOAK_HOME")
    return (Path(h) / "nous") if h else Path("data") / "nous"


def _vkey(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".") if x.isdigit())


def _manifest(folder: Path) -> dict:
    try:
        return json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError):
        return {}


def active_dir() -> Path:
    """The folder skills load from: a downloaded set if it's newer than the packaged one."""
    o, p = override_dir(), PRETRAINED
    mo, mp = _manifest(o), _manifest(p)
    if mo and _vkey(mo.get("nous", "0")) > _vkey(mp.get("nous", "0")) and _vkey(mo.get("min_app", "0")) <= _vkey(__version__):
        return o
    return p


def installed() -> str:
    return _manifest(active_dir()).get("nous", "0.0.0")


def check(timeout: float = 15.0) -> dict:
    """Ask GitHub which Nous version is on main."""
    out = {"installed": installed(), "latest": None, "update": False, "needs_app_update": False}
    try:
        with httpx.Client(timeout=timeout) as c:
            sha = c.get(f"https://api.github.com/repos/{REPO}/commits/main", headers={"Accept": "application/vnd.github.sha"}).text.strip()
            m = c.get(f"https://raw.githubusercontent.com/{REPO}/{sha}/cloakos/rl/pretrained/manifest.json").json()
    except Exception as e:
        out["error"] = f"couldn't check ({type(e).__name__}). Are you online?"
        return out
    out.update(latest=m.get("nous"), sha=sha, manifest=m)
    if _vkey(m.get("nous", "0")) > _vkey(out["installed"]):
        if _vkey(m.get("min_app", "0")) > _vkey(__version__):
            out["needs_app_update"] = True
        else:
            out["update"] = True
    return out


def update(timeout: float = 120.0) -> dict:
    """Download the latest policies, verify their hashes, and swap them in."""
    info = check()
    if not info.get("update"):
        return info
    m, sha = info["manifest"], info["sha"]
    dest = override_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="nous-", dir=dest.parent))
    try:
        with httpx.Client(timeout=timeout) as c:
            for skill, files in m["skills"].items():
                (tmp / skill).mkdir()
                for name, digest in files.items():
                    data = c.get(f"https://raw.githubusercontent.com/{REPO}/{sha}/cloakos/rl/pretrained/{skill}/{name}").content
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError(f"{skill}/{name} didn't match its checksum")
                    (tmp / skill / name).write_bytes(data)
        (tmp / "manifest.json").write_text(json.dumps(m, indent=1))
        old = dest.with_name(dest.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if dest.exists():
            dest.rename(old)
        tmp.rename(dest)
        shutil.rmtree(old, ignore_errors=True)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        info.update(update=False, error=f"update failed: {e}")
        return info
    info.update(installed=installed(), updated=True, update=False)
    info.pop("manifest", None)
    return info
