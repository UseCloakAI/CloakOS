"""Body: the single gateway to Vector's motors, animations and speaker.

Behaviors, teleop and (later) a learned policy all move the robot through here, so the
logger sees one consistent command stream. The continuous action space is Action
(wheel speeds + head/lift rates); one-shot things (anims, speech, turns) are logged as
discrete events.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from .robot import RobotLink

log = logging.getLogger(__name__)

MAX_WHEEL = 220.0   # mm/s
MAX_HEAD = 4.0      # rad/s
MAX_LIFT = 4.0      # rad/s


@dataclass
class Action:
    left_mmps: float = 0.0
    right_mmps: float = 0.0
    head_rate: float = 0.0
    lift_rate: float = 0.0

    def clamped(self) -> "Action":
        c = lambda v, m: max(-m, min(m, float(v)))  # noqa: E731
        return Action(c(self.left_mmps, MAX_WHEEL), c(self.right_mmps, MAX_WHEEL),
                      c(self.head_rate, MAX_HEAD), c(self.lift_rate, MAX_LIFT))

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in asdict(self).items()}


class Body:
    def __init__(self, link: RobotLink) -> None:
        self.link = link
        self.cmd = Action()
        self.source = "none"          # who sent the last command: behavior name, "teleop", "policy"
        self._discrete: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return self.link.connected and self.link.has_control

    @property
    def _r(self) -> Any:
        return self.link.robot

    def drain_discrete(self) -> list[dict[str, Any]]:
        out, self._discrete = self._discrete, []
        return out

    def _mark(self, kind: str, **data: Any) -> None:
        self._discrete.append({"kind": kind, "t": time.time(), "source": self.source, **data})

    async def _call(self, fut_factory, timeout: float = 10.0) -> Any:
        if not self.ready:
            return None
        try:
            return await self.link.call(fut_factory(), timeout=timeout)
        except Exception as e:
            log.debug("body call failed: %s", e)
            return None

    # ── continuous ─────────────────────────────────────────────

    async def apply(self, action: Action, source: str | None = None) -> None:
        """Send a continuous command. Only components that changed go over the wire."""
        a = action.clamped()
        prev, self.cmd = self.cmd, a
        if source:
            self.source = source
        if (a.left_mmps, a.right_mmps) != (prev.left_mmps, prev.right_mmps):
            await self._call(lambda: self._r.motors.set_wheel_motors(a.left_mmps, a.right_mmps, 400, 400), 2)
        if a.head_rate != prev.head_rate:
            await self._call(lambda: self._r.motors.set_head_motor(a.head_rate), 2)
        if a.lift_rate != prev.lift_rate:
            await self._call(lambda: self._r.motors.set_lift_motor(a.lift_rate), 2)

    async def drive(self, left: float, right: float, source: str | None = None) -> None:
        await self.apply(Action(left, right, self.cmd.head_rate, self.cmd.lift_rate), source)

    async def stop(self) -> None:
        self.cmd = Action()
        await self._call(lambda: self._r.motors.stop_all_motors(), 2)

    # ── discrete ───────────────────────────────────────────────

    def resolve_anim(self, key: str) -> str | None:
        """Exact trigger name, else a random trigger containing `key` (case-insensitive)."""
        triggers = self.link.anim_triggers
        if key in triggers:
            return key
        matches = [t for t in triggers if key.lower() in t.lower()]
        return random.choice(matches) if matches else None

    async def anim(self, *keys: str) -> str | None:
        """Play the first key that resolves to a trigger. Returns the trigger played."""
        for key in keys:
            trig = self.resolve_anim(key)
            if trig:
                self._mark("anim", trigger=trig)
                await self._call(lambda: self._r.anim.play_animation_trigger(trig, ignore_body_track=False), 15)
                return trig
        return None

    async def head_angle(self, deg: float) -> None:
        from anki_vector.util import degrees
        self._mark("head_angle", deg=deg)
        await self._call(lambda: self._r.behavior.set_head_angle(degrees(deg)), 5)

    async def lift_height(self, h: float) -> None:
        self._mark("lift_height", h=h)
        await self._call(lambda: self._r.behavior.set_lift_height(max(0.0, min(1.0, h))), 5)

    async def turn(self, deg: float, speed_dps: float = 90) -> None:
        from anki_vector.util import degrees
        self._mark("turn", deg=deg)
        await self._call(lambda: self._r.behavior.turn_in_place(degrees(deg), speed=degrees(speed_dps)), 10)

    async def straight(self, mm: float, speed_mmps: float = 80) -> None:
        from anki_vector.util import distance_mm, speed_mmps as spd
        self._mark("straight", mm=mm)
        await self._call(lambda: self._r.behavior.drive_straight(distance_mm(mm), spd(speed_mmps), should_play_anim=False), 15)

    async def say_wav(self, path: str, volume: int = 75, text: str = "") -> None:
        self._mark("say", text=text)
        await self._call(lambda: self._r.audio.stream_wav_file(path, volume), 30)
