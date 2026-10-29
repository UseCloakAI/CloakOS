"""LearnedPolicy: run a trained RL policy as one of the brain's modes ("learned").

Every policy runs at the brain's 10 Hz tick through Body, on the sim or the real robot, and
sees exactly what it saw in training: the same observation (envs.observe), ranges from the
live map (or the sim), and its own last command. The edge reflex runs robot-side (RobotLink),
the same reflex training models (envs.EdgeReflex).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..behaviors.base import Obs, Policy
from ..connection import Action
from .envs import RAY_FOV, RAY_MAX, Spec, edge_rays, observe

if TYPE_CHECKING:
    from ..connection import Body, RobotLink
    from ..mapping.mapper import Mapper
    from ..world import World

log = logging.getLogger(__name__)


def robot_snapshot(link: "RobotLink", world: "World") -> dict:
    """The sensor dict policies are trained on, from the sim or the real robot."""
    sim = link.sim
    if link.target == "sim" and sim is not None and sim.snapshot:
        return sim.snapshot
    w = world
    x, y, yaw = w.pose or (0.0, 0.0, 0.0)
    ax, ay, az = w.accel or (0.0, 0.0, 9800.0)
    g = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    return {"pos": [x / 1000, y / 1000, 0.0], "yaw": yaw, "left_mmps": w.left_mmps or 0.0,
            "right_mmps": w.right_mmps or 0.0, "prox_m": (w.proximity_mm / 1000) if w.proximity_mm else None,
            "cliff": w.cliff, "upright": az / g if g > 1 else 1.0,
            "pitch": math.asin(max(-1, min(1, ax / g))) if g > 1 else 0.0, "gyro": list(w.gyro or (0.0, 0.0, 0.0))}


RANGES_STATE: dict[str, Any] = {"source": None, "why": None}     # what the last ranges() call used, for the UI
_warned: set[str] = set()


def ranges(link: "RobotLink", mapper: "Mapper | None", s: dict, n: int, arena: Any = None) -> np.ndarray | None:
    """Ranges around the robot for "map" policies: the 3D map (parked, so normally off), else the sim,
    else the overhead Arena camera on the real robot. None (= all clear) when there's none of them,
    which is logged once and shown in the Arena status, not silently ignored."""
    if mapper is not None and mapper.map is not None:
        RANGES_STATE.update(source="map", why=None)
        return mapper.rays(s["pos"], s["yaw"], n, RAY_FOV, RAY_MAX)
    sim = link.sim
    if link.target == "sim" and sim is not None:
        RANGES_STATE.update(source="sim", why=None)
        return sim.rays(n, RAY_FOV, RAY_MAX)
    why = "open the Arena page so the desk camera can see Vector" if arena is None else arena.ranges_status()
    if why is None:
        r = arena.ranges(n, RAY_FOV, RAY_MAX)
        if r is not None:
            RANGES_STATE.update(source="arena", why=None)
            return r
        why = "Arena gave no ranges"
    RANGES_STATE.update(source=None, why=why)
    if why not in _warned:
        _warned.add(why)
        log.warning("no ranges for a map driver on the real robot (it sees all clear): %s", why)
    return None


class NumpyPolicy:
    """A trained PPO actor run with numpy: tanh hidden layers, linear output, clipped."""

    def __init__(self, path: Path) -> None:
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("w")])
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        x = np.asarray(obs, np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1, 1), None


class LearnedPolicy(Policy):
    name = "learned"

    def __init__(self, body: "Body", link: "RobotLink") -> None:
        self.body, self.link = body, link
        self.mapper: "Mapper | None" = None                # set by the brain once the mapper exists
        self.arena_src: Any = None                         # () -> the Arena or None, set by the brain
        self.model: Any = None
        self.spec: Spec | None = None
        self.policy_name: str | None = None
        self.goal: tuple[float, float] | None = None       # metres, in the robot's pose frame
        self.last_action = np.zeros(2, np.float32)
        self.status = "no policy loaded"
        self.pixels = None

    @property
    def current(self) -> str:   # shown as the behavior
        return f"learned:{self.spec.task}" if self.spec else "learned"

    def load(self, folder: str | Path) -> None:
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        self.pixels = None
        self._frames: tuple[int, np.ndarray | None, np.ndarray | None] = (-1, None, None)
        if meta["sensors"] == "pixels":                 # the desk student: camera + body sensors
            from .pixelnet import PixelNet
            self.pixels, self.model = PixelNet(folder / "policy.npz"), None
            self.spec = Spec("desk", "robot", meta["control_hz"], meta["max_speed"])
        else:
            self.model = NumpyPolicy(folder / "policy.npz")
            self.spec = Spec(meta["task"], meta["sensors"], meta["control_hz"], meta["max_speed"], n_rays=meta.get("n_rays", 9))
        self.policy_name = folder.name
        self.status = f"loaded {folder.name}"
        log.info("learned policy: %s (%s, %s sensors)", folder.name, self.spec.task, self.spec.sensors)

    def deployable(self) -> str | None:
        """Why this policy can't drive the current target, or None if it can."""
        if self.spec is None:
            return "no policy loaded"
        if self.spec.task == "desk" and self.pixels is None and self.link.target != "sim":
            a = self.arena_src() if self.arena_src else None    # the overhead camera can stand in for exact edges
            why = "open the Arena" if a is None else a.ranges_status()
            if why:
                return f"the desk teacher needs exact edges: the sim, or the Arena on the real desk ({why})"
        if self.spec.task == "goto" and self.goal is None:
            return "set a goal first"
        return None

    # ── Policy API ─────────────────────────────────────────────

    async def start(self) -> None:
        self.last_action = np.zeros(2, np.float32)

    async def stop(self) -> None:
        await self.body.stop()

    async def step(self, obs: Obs) -> None:
        why = self.deployable()
        if why:
            self.status = why
            await self.body.drive(0, 0, "learned")
            return
        s = robot_snapshot(self.link, obs.world)
        if self.spec.task == "goto" and math.dist(s["pos"][:2], self.goal) < 0.06:
            self.status = "arrived"
            await self.body.drive(0, 0, "learned")
            return
        if self.pixels is not None:
            a = self._pixel_action(obs, s)
            if a is None:
                self.status = "waiting for the camera"
                await self.body.drive(0, 0, "learned")
                return
        else:
            rays = None
            if self.spec.task == "desk" and self.link.target == "sim" and self.link.sim and self.link.sim.scene_info:     # teacher: exact sim ranges
                sim = self.link.sim
                rays = np.minimum(sim.rays(self.spec.n_rays, RAY_FOV, RAY_MAX), edge_rays(s, sim.scene_info.room, self.spec.n_rays))
            elif self.spec.sensors == "map":
                rays = ranges(self.link, self.mapper, s, self.spec.n_rays, self.arena_src() if self.arena_src else None)
                if rays is None:            # all clear would be a guess: hold still until the ranges are back
                    self.status = f"waiting for ranges ({RANGES_STATE['why']})"
                    await self.body.drive(0, 0, "learned")
                    return
            a, _ = self.model.predict(observe(self.spec, s, self.goal, rays, self.last_action), deterministic=True)
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        self.last_action = a
        self.status = "driving"
        await self.body.apply(Action(float(a[0]) * self.spec.max_speed, float(a[1]) * self.spec.max_speed,
                                     self.body.cmd.head_rate, self.body.cmd.lift_rate), source="learned")

    def _pixel_action(self, obs: Obs, s: dict) -> np.ndarray | None:
        from .pixelnet import prepare

        w = obs.world
        if w.frame is None:
            return None
        fid, cur, prev = self._frames
        if w.frame_id != fid or cur is None:
            prev, cur = cur, prepare(w.frame)
            self._frames = (w.frame_id, cur, prev)
        return self.pixels.act(cur, prev if prev is not None else cur, observe(self.spec, s, None, None, self.last_action))
