"""Body: the single gateway to Vector's motors, screen and speaker.

Behaviors, teleop and (later) a learned policy all move the robot through here, so the
logger sees one consistent command stream. The continuous action space is Action
(wheel speeds + head/lift rates); one-shot things (emotes, turns, speech) are logged as
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
        self.emotes: Any = None       # expression.Emotes, attached by the brain
        self._discrete: list[dict[str, Any]] = []
        self._gen = -1
        self.face_ok, self.face_error, self._face_warned = 0, None, 0.0     # eyes frames Vector accepted / last error
        self.frozen = False           # freeze mode: no motion at all until unfrozen                # link.control_gen our last command went out under

    # while frozen only these get through: the screen, sound, and stopping
    FROZEN_OK = {"show_face", "play_pcm", "stop_all"}

    @property
    def ready(self) -> bool:
        return self.link.connected and self.link.has_control

    def drain_discrete(self) -> list[dict[str, Any]]:
        out, self._discrete = self._discrete, []
        return out

    def _mark(self, kind: str, **data: Any) -> None:
        self._discrete.append({"kind": kind, "t": time.time(), "source": self.source, **data})

    async def _do(self, method: str, *args: Any) -> Any:
        """Call a VectorClient method if we hold control; failures are logged, never raised."""
        if not self.ready or (self.frozen and method not in self.FROZEN_OK):
            return None
        try:
            out = await getattr(self.link.client, method)(*args)
            if method == "show_face":
                self.face_ok, self.face_error = self.face_ok + 1, None
            return out
        except Exception as e:
            if method == "show_face":                   # the eyes not showing is worth knowing about
                self.face_error = f"{type(e).__name__}: {e}"[:200]
                if time.time() - self._face_warned > 30:
                    self._face_warned = time.time()
                    log.warning("Vector didn't take the eyes frame: %s", self.face_error)
            else:
                log.debug("body %s failed: %s", method, e)
            return None

    # ── continuous ─────────────────────────────────────────────

    async def apply(self, action: Action, source: str | None = None) -> None:
        """Send a continuous command. Only components that changed go over the wire."""
        a = action.clamped()
        prev, self.cmd = self.cmd, a
        if source:
            self.source = source
        if not self.ready or self.frozen:
            return
        if self._gen != self.link.control_gen:   # new control session: the robot was stopped, resend everything
            self._gen, prev, self._sent = self.link.control_gen, Action(), None
        if getattr(self.link, "edge_events", 0) != getattr(self, "_edges", 0):   # the edge reflex moved the wheels
            self._edges, self._sent = self.link.edge_events, None
        wheels = (a.left_mmps, a.right_mmps)
        if getattr(self.link.world, "cliff", False):  # edge under the nose: no forward (at override priority
            wheels = (min(0.0, wheels[0]), min(0.0, wheels[1]))   # Vector's own edge stop is off)
        if wheels != getattr(self, "_sent", None):
            await self._do("drive_wheels", wheels[0], wheels[1], 400)
            self._sent = wheels
        if a.head_rate != prev.head_rate:
            await self._do("move_head", a.head_rate)
        if a.lift_rate != prev.lift_rate:
            await self._do("move_lift", a.lift_rate)

    async def drive(self, left: float, right: float, source: str | None = None) -> None:
        await self.apply(Action(left, right, self.cmd.head_rate, self.cmd.lift_rate), source)

    async def stop(self) -> None:
        self.cmd = Action()
        self._sent = (0.0, 0.0)
        await self._do("stop_all")

    # ── discrete ───────────────────────────────────────────────

    async def emote(self, name: str) -> None:
        """Play a custom emote (expression/emotes.py), sometimes blended with a stock animation."""
        self._mark("emote", name=name)
        if self.emotes is not None and self.ready:
            await self.emotes.play(name, self)

    def resolve_anim(self, key: str) -> str | None:
        """Exact stock trigger name, else a random trigger containing `key` (case-insensitive)."""
        triggers = self.link.anim_triggers
        if key in triggers:
            return key
        matches = [t for t in triggers if key.lower() in t.lower()]
        return random.choice(matches) if matches else None

    async def anim(self, *keys: str) -> str | None:
        """Play Vector's own animation: the first key that resolves to a trigger."""
        for key in keys:
            trig = self.resolve_anim(key)
            if trig:
                self._mark("anim", trigger=trig)
                await self._do("play_trigger", trig)
                return trig
        return None

    async def head_angle(self, deg: float, speed: float = 10.0) -> None:
        self._mark("head_angle", deg=deg)
        await self._do("head_angle", deg, speed)

    async def lift_height(self, h: float, speed: float = 10.0) -> None:
        self._mark("lift_height", h=h)
        await self._do("lift_height", h, speed)

    async def turn(self, deg: float, speed_dps: float = 90) -> None:
        self._mark("turn", deg=deg)
        await self._do("turn", deg, speed_dps)

    async def straight(self, mm: float, speed_mmps: float = 80) -> None:
        if mm > 0 and getattr(self.link.world, "cliff", False):
            return                                      # never roll forward over an edge
        self._mark("straight", mm=mm)
        await self._do("straight", mm, speed_mmps)

    async def go_home(self) -> bool:
        self._mark("go_home")
        return bool(await self._do("drive_on_charger"))

    async def leave_home(self) -> None:
        self._mark("leave_home")
        await self._do("drive_off_charger")

    async def face(self, rgb565: bytes, duration_ms: int) -> None:
        """Draw a frame on the screen. Not logged: the eye renderer calls this ~10x/s."""
        await self._do("show_face", rgb565, duration_ms)

    async def play_pcm(self, pcm: bytes, volume: int = 75, text: str = "", rate: int = 16000) -> None:
        self._mark("say" if text else "sound", text=text)
        if text:                                     # sound effects wait while it talks
            self.speaking_until = time.time() + len(pcm) / 2 / rate + 0.2
        master = self.link.volume
        if master > 0:
            await self._do("play_pcm", pcm, rate, round(volume * master / 100))
