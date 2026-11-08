"""Write cloakos/rl/pretrained/manifest.json: the shipped Nous model version plus a hash of every file.

    python scripts/nous_manifest.py 0.0.1
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "cloakos" / "rl" / "pretrained"


EPOCH = 1   # bumped when Nous restarts from scratch: older downloaded models are ignored


def main(version: str, min_app: str = "0.2.0") -> None:
    skills = {}
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        skills[d.name] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.iterdir()) if f.is_file()}
    (ROOT / "manifest.json").write_text(json.dumps({"nous": version, "epoch": EPOCH, "min_app": min_app, "skills": skills}, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
