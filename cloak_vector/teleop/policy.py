"""TeleopPolicy: you drive. Input arrives from the web UI (keyboard or browser Gamepad API)
as analog axes in -1..1; this turns them into the same Action the logger records.

Dead-man: if no input arrives for `deadman_s`, the bot stops.
"""
from __future__ import annotations

import time

from ..connection import Action, Body
from ..behaviors.base import Obs, Policy


class TeleopPolicy(Policy):
    name = "teleop"
    current = "teleop"

    def __init__(self, cfg: dict, body: Body) -> None:
        c = cfg.get("teleop", {}) or {}
        self.max_speed = float(c.get("max_speed_mmps", 150))
        self.boost = float(c.get("boost", 1.4))
        self.turn_gain = float(c.get("turn_gain", 0.7))
        self.head_speed = float(c.get("head_rad_s", 1.5))
        self.lift_speed = float(c.get("lift_rad_s", 2.0))
        self.deadman_s = float(c.get("deadman_s", 0.5))
        self.body = body
        self.axes = {"throttle": 0.0, "steer": 0.0, "head": 0.0, "lift": 0.0, "boost": 0.0}
        self._last_input = 0.0

    def input(self, **axes: float) -> None:
        for k, v in axes.items():
            if k in self.axes:
                self.axes[k] = max(-1.0, min(1.0, float(v)))
        self._last_input = time.time()

    def action(self) -> Action:
        if time.time() - self._last_input > self.deadman_s:
            return Action()
        a = self.axes
        v = a["throttle"] * self.max_speed * (self.boost if a["boost"] > 0 else 1.0)
        w = a["steer"] * self.max_speed * self.turn_gain
        return Action(v + w, v - w, a["head"] * self.head_speed, a["lift"] * self.lift_speed)

    async def step(self, obs: Obs) -> None:
        await self.body.apply(self.action(), source="teleop")

    async def stop(self) -> None:
        self.axes = dict.fromkeys(self.axes, 0.0)
        await self.body.stop()
