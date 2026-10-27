"""GoToPolicy: drive to something Vector remembers ("go to the mug").

Uses the trained goto skill when it ships (it steers around obstacles); otherwise a simple
point-and-drive controller that stops for anything close in front. Stops a little short of
the object, says when it got there, gives up if stuck or out of time.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from .base import Obs, Policy

if TYPE_CHECKING:
    from ..bus import Bus
    from ..connection import Body
    from ..rl.skills import Skills

ARRIVE_M = 0.07
STANDOFF_M = 0.09          # stop this far from the object's centre
TIMEOUT_S = 60.0
STUCK_S = 10.0


class GoToPolicy(Policy):
    name = "goto"

    def __init__(self, body: "Body", skills: "Skills", bus: "Bus") -> None:
        self.body, self.skills, self.bus = body, skills, bus
        self.target: dict | None = None
        self.goal: tuple[float, float] | None = None
        self.status = "idle"
        self._t0 = 0.0
        self._best = math.inf
        self._best_t = 0.0

    @property
    def current(self) -> str:
        return f"going to {self.target['label']}" if self.target else "goto"

    def set_target(self, obj: dict, here: tuple[float, float]) -> None:
        """Aim for a point just short of the object, on the near side."""
        ox, oy = obj["pos"][0], obj["pos"][1]
        dx, dy = ox - here[0], oy - here[1]
        d = math.hypot(dx, dy) or 1.0
        stand = min(STANDOFF_M, d * 0.5)
        self.target, self.goal = obj, (ox - dx / d * stand, oy - dy / d * stand)
        self.status = "driving"

    async def start(self) -> None:
        self._t0 = self._best_t = time.time()
        self._best = math.inf
        if self.skills.has("goto"):
            self.skills.skills["goto"].reset()

    async def stop(self) -> None:
        await self.body.stop()

    async def step(self, obs: Obs) -> None:
        if self.goal is None or self.status in ("arrived", "failed"):
            await self.body.drive(0, 0, "goto")
            return
        s = self.skills.snapshot()
        dist = math.dist(s["pos"][:2], self.goal)
        now = time.time()
        if dist < self._best - 0.02:
            self._best, self._best_t = dist, now
        if dist < ARRIVE_M:
            return await self._finish("arrived")
        if now - self._t0 > TIMEOUT_S or now - self._best_t > STUCK_S:
            return await self._finish("failed")
        if self.skills.has("goto"):
            left, right, _ = self.skills.act("goto", self.goal)
        else:
            left, right = self._simple(s)
        await self.body.drive(left, right, "goto")

    def _simple(self, s: dict) -> tuple[float, float]:
        """No trained skill: turn toward the goal, roll forward, stop for close things."""
        ang = math.atan2(self.goal[1] - s["pos"][1], self.goal[0] - s["pos"][0]) - s["yaw"]
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        turn = max(-1.0, min(1.0, 1.5 * ang))
        prox = s.get("prox_m")
        fwd = 0.0 if abs(ang) > 0.6 or (prox is not None and prox < 0.07) or s.get("cliff") else 90.0
        return fwd - 60 * turn, fwd + 60 * turn

    async def _finish(self, how: str) -> None:
        self.status = how
        await self.body.drive(0, 0, "goto")
        self.bus.emit(f"goto.{how}", target=self.target["label"] if self.target else None,
                      id=self.target["id"] if self.target else None)
