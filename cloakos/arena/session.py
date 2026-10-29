"""Arena: the real desk, watched by a camera above it (an iPhone via Continuity Camera works).

The page does the seeing: it reads the camera in the browser, finds Vector's gold top (and the
dot, for which way it faces), and maps that through the desk corners you clicked into desk
coordinates. It posts the pose here about 10 times a second.

This side keeps the desk setup and scores real runs, the same marks as the sim exam:
desk covered per minute, edge saves (times the edge reflex had to step in), falls.
A fall ends the run (someone has to put Vector back).

Desk frame: origin at the front-left corner, x to the right, y away from you, metres.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..app import Brain

CELL = 0.05
LOST_S = 3.0          # no sighting this long: tracking is lost (paused, not a fall)


class Arena:
    def __init__(self, brain: "Brain", data_dir: str | Path = "data") -> None:
        self.brain = brain
        self.file = Path(data_dir) / "arena.json"
        self.runs_file = Path(data_dir) / "arena_runs.jsonl"
        try:
            self.config: dict[str, Any] = json.loads(self.file.read_text())
        except (OSError, ValueError):
            self.config = {}
        self.pose: dict[str, Any] | None = None
        self.task: asyncio.Task | None = None
        self.run: dict[str, Any] | None = None

    # ── setup ──────────────────────────────────────────────────

    def save_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Corners (image points), colours and the desk size in cm. The size also goes to training:
        half the sim desks are then your desk's size (CLOAK_DESK)."""
        self.config.update({k: v for k, v in cfg.items() if k in ("corners", "size", "gold", "dot", "flip", "tol", "camera")})
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.config, indent=1))
        size = self.config.get("size")
        if size and all(float(v) > 0 for v in size):
            desk = f"{round(float(size[0]))}x{round(float(size[1]))}"
            os.environ["CLOAK_DESK"] = desk                   # training runs started from now on
            try:
                from ..web.setup_api import env_path, write_env
                write_env(env_path(), {"CLOAK_DESK": desk})
            except Exception:
                pass
        return self.config

    @property
    def size_m(self) -> tuple[float, float] | None:
        s = self.config.get("size")
        return (float(s[0]) / 100, float(s[1]) / 100) if s else None

    def update_pose(self, p: dict[str, Any]) -> None:
        self.pose = {**p, "t": time.time()}

    # ── scored runs ────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self, policy: str | None, seconds: float) -> str | None:
        b = self.brain
        if not self.size_m or not self.config.get("corners"):
            return "Set up the desk first (corners and size)."
        if not self.pose or time.time() - self.pose["t"] > LOST_S:
            return "The camera can't see Vector right now."
        if b.estop:
            return "Release the emergency stop first."
        if self.running:
            return "A run is already going."
        if policy:
            try:
                await asyncio.to_thread(b.learned.load, b.trainer.root / policy)
            except Exception as e:
                return f"Couldn't load {policy}: {e}"
            why = b.learned.deployable()
            if why:
                return why
            if b.mode == "learned":
                await b.set_mode("idle")
            await b.set_mode("learned")
        self.run = {"policy": policy or None, "driver": policy or f"you ({b.mode})", "target": b.link.target,
                    "start": time.time(), "seconds": float(seconds), "cells": [], "falls": 0, "edge_saves": 0,
                    "lost_s": 0.0, "status": "running", "path": []}
        self.task = asyncio.create_task(self._watch(policy))
        return None

    async def stop(self, why: str = "stopped") -> None:
        if self.running:
            self.task.cancel()
            await self._end(why)

    async def _watch(self, policy: str | None) -> None:
        r, link = self.run, self.brain.link
        edges0 = getattr(link, "edge_events", 0)
        cells: set[tuple[int, int]] = set()
        w, d = self.size_m
        try:
            while time.time() - r["start"] < r["seconds"]:
                await asyncio.sleep(0.1)
                if policy and self.brain.mode != "learned":
                    return await self._end("mode changed")
                r["edge_saves"] = getattr(link, "edge_events", 0) - edges0
                p = self.pose
                if not p or time.time() - p["t"] > LOST_S:
                    r["lost_s"] += 0.1
                    continue
                if not p.get("ok"):
                    continue
                x, y = float(p["x"]), float(p["y"])
                if x < -0.02 or y < -0.02 or x > w + 0.02 or y > d + 0.02:     # its middle is past the edge
                    r["falls"] += 1
                    return await self._end("fell")
                cells.add((math.floor(x / CELL), math.floor(y / CELL)))
                r["cells"] = len(cells)
                if not r["path"] or math.dist(r["path"][-1][:2], (x, y)) > 0.01:
                    r["path"].append((round(x, 3), round(y, 3), round(float(p.get("yaw", 0.0)), 2)))
                    del r["path"][:-3000]
        except asyncio.CancelledError:
            return
        await self._end("done")

    async def _end(self, why: str) -> None:
        r = self.run
        if r is None or r.get("status") != "running":
            return
        r["status"] = why
        r["end"] = time.time()
        if r.get("policy") and self.brain.mode == "learned":
            await self.brain.set_mode("idle")
        res = self.summary()
        try:
            with self.runs_file.open("a") as f:
                f.write(json.dumps({**res, "path": r["path"]}) + "\n")
            if r.get("policy"):                          # next to the policy, for the Training page
                pf = self.brain.trainer.root / r["policy"] / "real_exams.json"
                old = json.loads(pf.read_text()) if pf.exists() else []
                pf.write_text(json.dumps((old + [res])[-20:], indent=1))
        except (OSError, ValueError):
            pass
        self.task = None

    def summary(self) -> dict[str, Any] | None:
        r = self.run
        if r is None:
            return None
        secs = max(1e-6, (r.get("end") or time.time()) - r["start"])
        mins = secs / 60
        w, d = self.size_m or (1, 1)
        total = (w / CELL) * (d / CELL)
        return {"policy": r["policy"], "driver": r["driver"], "status": r["status"], "when": r["start"],
                "seconds": round(secs), "planned_s": r["seconds"], "cells": r["cells"],
                "covered": round(min(1.0, r["cells"] / total), 3), "cells_per_min": round(r["cells"] / mins, 1),
                "edge_saves": r["edge_saves"], "edge_saves_per_min": round(r["edge_saves"] / mins, 2),
                "falls": r["falls"], "lost_s": round(r["lost_s"], 1), "target": r["target"]}

    def recent(self, n: int = 10) -> list[dict]:
        try:
            lines = self.runs_file.read_text().splitlines()[-n:]
        except OSError:
            return []
        out = []
        for x in reversed(lines):
            try:
                out.append({k: v for k, v in json.loads(x).items() if k != "path"})
            except ValueError:
                pass
        return out
