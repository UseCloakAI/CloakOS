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
POINT_ARRIVE_M = 0.025     # Arena points: the overhead camera is accurate, so be accurate
SLOW_M = 0.12              # final approach: slow down inside this
SLOW_MMPS = 25.0
POINT_MAX_MMPS = 70.0      # accuracy over speed: Wi-Fi lag makes fast driving overshoot
EDGE_STOP_M = 0.03         # a buffered ray ahead shorter than this: stop
STANDOFF_M = 0.09          # stop this far from the object's centre
TIMEOUT_S = 60.0
STUCK_S = 10.0
NUDGE_MMPS = 40.0          # heading check: a slow, straight few cm forward
NUDGE_S = 5.0              # ...given up (and go-to refused) if driving hasn't settled it by then
NUDGE_EDGE_M = 0.08        # not this close to any desk edge (the heading, so which edge is ahead, isn't known yet)


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
        self._nudge_t: float | None = None
        self.frame = "odo"                    # "arena": the goal is in Arena desk metres
        self.arena_src = None                 # () -> the Arena or None, set by the brain
        self.teacher: dict | None = None      # the last decision, for recordings

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
        self.frame = "odo"
        self.status = "driving"

    def set_point(self, x: float, y: float, label: str | None = None) -> None:
        """A point on the desk, in Arena coordinates (metres from the front-left corner)."""
        self.target = {"id": None, "label": label or f"point {x * 100:.0f}, {y * 100:.0f} cm", "pos": [x, y, 0.0]}
        self.goal, self.frame, self.status = (x, y), "arena", "driving"

    async def start(self) -> None:
        self._t0 = self._best_t = time.time()
        self._best = math.inf
        self._nudge_t = None
        if self.skills.has("goto"):
            self.skills.skills["goto"].reset()

    async def stop(self) -> None:
        await self.body.stop()

    async def step(self, obs: Obs) -> None:
        if self.goal is None or self.status.startswith(("arrived", "failed")):
            await self.body.drive(0, 0, "goto")
            return
        if self.frame == "arena":
            return await self._step_arena()
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

    async def _step_arena(self) -> None:
        """Drive to an Arena point: latency-aware desk pose, hold still when it's stale, slow at the end."""
        from ..rl.envs import RAY_FOV, RAY_MAX, goal_features
        a = self.arena_src() if self.arena_src else None
        base = self.skills.snapshot()
        s = a.snapshot(base) if a is not None else None
        now = time.time()
        if a is not None and not getattr(a, "heading_confirmed", True):
            return await self._nudge(a, now)
        self._nudge_t = None
        if s is None:                         # no fresh overhead pose: stopping beats guessing
            self.teacher = None
            self._best_t = now                # waiting isn't being stuck
            await self.body.drive(0, 0, "goto")
            self.status = "waiting for the desk camera"
            return
        self.status = "driving"
        if a is not None:
            a.route = {"target": list(self.goal), "charger": list(self.goal), "phase": "goto"}
        dist = math.dist(s["pos"][:2], self.goal)
        if dist < self._best - 0.01:
            self._best, self._best_t = dist, now
        if dist < POINT_ARRIVE_M:
            return await self._finish("arrived")
        if now - self._t0 > TIMEOUT_S * 2 or now - self._best_t > STUCK_S:
            return await self._finish("failed")
        rays = a.ranges(9, RAY_FOV, RAY_MAX)
        if self.skills.has("goto"):
            left, right, _ = self.skills.act("goto", self.goal, s)
            src = "skill"
            if left + right < 0:              # no reversing on the desk: Vector has no cliff sensors behind
                w = (right - left) / 2
                left, right = -w, w
        else:
            left, right = self._point_controller(s, dist)
            src = "controller"
        # never drive at an edge: the middle rays are already shortened by the stopping distance,
        # and the goal is always inside the desk, so a short ray ahead means the edge is closer than it
        if rays is not None and (left + right) > 0 and float(min(rays[3:6])) < min(EDGE_STOP_M, dist):
            left = right = 0.0
            self.status = "edge ahead: stopped"
        self.teacher = {"task": "goto", "src": src, "goal": list(self.goal), "goal_rel": goal_features(s, self.goal),
                        "action": [left / 220.0, right / 220.0]}
        await self.body.drive(left, right, "goto")

    async def _nudge(self, a, now: float) -> None:
        """The camera can mistake Vector's back for its front. Before steering, roll straight forward a
        few cm: the way it moves settles which end is which (the Arena flips it if needed)."""
        p = a.pose
        fresh = p and p.get("ok") and p.get("x") is not None and now - p["t"] < 0.5
        if self._nudge_t is None:
            self._nudge_t = now
        self._best_t = now
        if not fresh:
            await self.body.drive(0, 0, "goto")
            self.status = "heading check: waiting for the desk camera"
            if now - self._nudge_t > NUDGE_S * 2:
                return await self._finish("failed", "the desk camera lost Vector during the heading check")
            return
        x, y, sz = float(p["x"]), float(p["y"]), a.size_m
        if not sz or min(x, y, sz[0] - x, sz[1] - y) < NUDGE_EDGE_M:
            return await self._finish("failed", "too close to a desk edge to check its heading safely: move it in a bit")
        if now - self._nudge_t > NUDGE_S:
            return await self._finish("failed", "couldn't confirm which way Vector faces by driving")
        self.status = "heading check: rolling forward a few cm"
        self.teacher = None
        await self.body.drive(NUDGE_MMPS, NUDGE_MMPS, "goto")

    def _point_controller(self, s: dict, dist: float) -> tuple[float, float]:
        """No trained skill: turn toward the point, gentle gains (commands land ~0.4 s late), slow at the end."""
        ang = math.atan2(self.goal[1] - s["pos"][1], self.goal[0] - s["pos"][0]) - s["yaw"]
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        if abs(ang) > 0.8:                                   # well off: turn in place, slowly
            w = 35.0 if ang > 0 else -35.0
            return -w, w
        v = min(POINT_MAX_MMPS, SLOW_MMPS + 300 * max(0.0, dist - SLOW_M)) if dist > SLOW_M else SLOW_MMPS
        v *= max(0.2, math.cos(ang))
        w = max(-30.0, min(30.0, 60 * ang))
        return v - w, v + w

    def _simple(self, s: dict) -> tuple[float, float]:
        """No trained skill: turn toward the goal, roll forward, stop for close things."""
        ang = math.atan2(self.goal[1] - s["pos"][1], self.goal[0] - s["pos"][0]) - s["yaw"]
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        turn = max(-1.0, min(1.0, 1.5 * ang))
        prox = s.get("prox_m")
        fwd = 0.0 if abs(ang) > 0.6 or (prox is not None and prox < 0.07) or s.get("cliff") else 90.0
        return fwd - 60 * turn, fwd + 60 * turn

    async def _finish(self, how: str, why: str | None = None) -> None:
        self.status = f"{how}: {why}" if why else how
        self.why = why
        self.teacher = None
        a = self.arena_src() if self.arena_src else None
        if a is not None and self.frame == "arena":
            a.route = None
        await self.body.drive(0, 0, "goto")
        self.bus.emit(f"goto.{how}", why=why, target=self.target["label"] if self.target else None,
                      id=self.target["id"] if self.target else None)
