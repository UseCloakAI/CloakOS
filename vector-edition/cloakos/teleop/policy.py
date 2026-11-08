"""TeleopPolicy: you drive. Input arrives from the web UI (keyboard or browser Gamepad API)
as analog axes in -1..1; this turns them into the same Action the logger records.

Dead-man: if no input arrives for `deadman_s`, the bot stops.

Home pose: a second after you stop moving the head or lift, they go back to one fixed pose (head
a little down, lift down), so the camera always sees the desk the same way. That keeps teleop
recordings consistent for training, and the view predictable. teleop.head_home_deg (None = off).
"""
from __future__ import annotations

import asyncio
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
        home = c.get("head_home_deg", -5.0)               # the middle of the head angles desk training covers
        self.head_home = None if home is None else float(home)
        self.home_after_s = float(c.get("home_after_s", 1.0))
        self._posed = 0.0                                 # when head/lift were last moved by hand (0 = at home)

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
        act = self.action()
        if act.head_rate or act.lift_rate:
            self._posed = time.time()
        elif self._posed and time.time() - self._posed > self.home_after_s:
            self._posed = 0.0
            self._go_home_pose()
        await self.body.apply(act, source="teleop")

    async def start(self) -> None:
        self._go_home_pose()

    def _go_home_pose(self) -> None:
        if self.head_home is None:
            return
        async def pose() -> None:
            try:
                await asyncio.gather(self.body.head_angle(self.head_home), self.body.lift_height(0.0))
            except Exception:
                pass                                       # not connected, or halted: nothing to straighten
        asyncio.ensure_future(pose())

    async def stop(self) -> None:
        self.axes = dict.fromkeys(self.axes, 0.0)
        await self.body.stop()
