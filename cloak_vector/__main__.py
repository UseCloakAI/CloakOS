"""python -m cloak_vector [--mode auto|teleop|idle] [--config config.yaml] [--no-web]"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .app import MODES, Brain
from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="cloak-vector", description="Cloak for Vector")
    ap.add_argument("--mode", choices=MODES, default=None, help="starting mode (default: config `mode`)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-web", action="store_true", help="headless: no dashboard")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("grpc", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    brain = Brain(cfg)
    asyncio.run(brain.run(mode=args.mode or cfg.get("mode", "auto"), web=not args.no_web))
    if brain.restart:   # the setup wizard saved new settings: start fresh with them
        os.execv(sys.executable, [sys.executable, "-m", "cloak_vector", *sys.argv[1:]])


if __name__ == "__main__":
    main()
