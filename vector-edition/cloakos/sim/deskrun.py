"""Desk run: let a trained desk policy drive the sim desk you built and score it live.

Same marks as the desk exam (desk covered per minute, bumps per minute, falls), but on your
desk, with things you add or take away while it drives. A fall puts Vector back in the middle.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..app import Brain

CELL = 0.05


class DeskRun:
    def __init__(self, brain: "Brain") -> None:
        self.brain = brain
        self.task: asyncio.Task | None = None
        self.policy: str | None = None
        self.t0 = 0.0
        self.cells: set[tuple[int, int]] = set()
        self.bumps = self.falls = 0
        self.last: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self, policy: str) -> str | None:
        b = self.brain
        sim = b.link.sim
        if b.link.target != "sim" or sim is None:
            return "Switch to Sim first."
        if b.estop:
            return "Release the emergency stop first."
        if sim.world != "desk":
            return "Load a desk first."
        folder = b.trainer.root / policy
        try:
            await asyncio.to_thread(b.learned.load, folder)
        except Exception as e:
            return f"Couldn't load {policy}: {e}"
        if b.learned.spec is None or not b.learned.spec.task.startswith("desk"):
            return f"{policy} isn't a desk policy."
        self.stop_task()
        self.policy, self.t0, self.cells, self.bumps, self.falls, self.last = policy, time.monotonic(), set(), 0, 0, None
        if b.mode == "learned":
            await b.set_mode("idle")
        await b.set_mode("learned")
        self.task = asyncio.create_task(self._watch())
        return None

    def stop_task(self) -> None:
        if self.running:
            self.task.cancel()

    async def stop(self) -> None:
        if self.running:
            self.stop_task()
            self._finish()
            if self.brain.mode == "learned":
                await self.brain.set_mode("idle")

    def _finish(self) -> None:
        self.last = self.status()
        self.last["done"] = True
        self.task = None

    async def _watch(self) -> None:
        bumping = False
        try:
            while True:
                await asyncio.sleep(0.1)
                sim = self.brain.link.sim
                if sim is None or self.brain.mode != "learned" or sim.world != "desk":
                    break
                s, sc = sim.snapshot or {}, sim.scene_info
                pos = s.get("pos")
                if not pos or sc is None:
                    continue
                if pos[2] < sc.floor_z - 0.03 or (s.get("upright", 1) < 0.3):
                    self.falls += 1
                    with sim._lock:
                        sim.place_robot(0.0, 0.0, float(time.monotonic() * 2.3 % (2 * math.pi)))
                    continue
                self.cells.add((math.floor(pos[0] / CELL), math.floor(pos[1] / CELL)))
                if s.get("bump") and not bumping:
                    self.bumps += 1
                bumping = bool(s.get("bump"))
        except asyncio.CancelledError:
            return
        self._finish()

    def status(self) -> dict[str, Any] | None:
        if not self.running:
            return self.last
        sim = self.brain.link.sim
        sc = sim.scene_info if sim else None
        mins = max(1e-6, (time.monotonic() - self.t0) / 60)
        total = (sc.room[0] * 2 / CELL) * (sc.room[1] * 2 / CELL) if sc else 0
        return {"policy": self.policy, "running": True, "seconds": round(mins * 60),
                "cells": len(self.cells), "covered": round(min(1.0, len(self.cells) / total), 3) if total else None,
                "cells_per_min": round(len(self.cells) / mins, 1), "bumps": self.bumps,
                "bumps_per_min": round(self.bumps / mins, 2), "falls": self.falls,
                "status": self.brain.learned.status}
