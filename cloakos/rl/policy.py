"""LearnedPolicy: run a trained RL policy as one of the brain's modes ("learned").

goto / explore run at the brain's 10 Hz tick through Body, so they work on the sim and on
the real robot (robot-sensor policies only). balance needs 50 Hz, faster than Wi-Fi allows,
so it runs inside the sim's physics thread and is sim-only.
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
from .envs import Spec, observe

if TYPE_CHECKING:
    from ..connection import Body, RobotLink

log = logging.getLogger(__name__)


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
        self.model: Any = None
        self.spec: Spec | None = None
        self.policy_name: str | None = None
        self.goal: tuple[float, float] | None = None       # metres, in the robot's pose frame
        self.last_action = np.zeros(2, np.float32)
        self.status = "no policy loaded"

    @property
    def current(self) -> str:   # shown as the behavior
        return f"learned:{self.spec.task}" if self.spec else "learned"

    def load(self, folder: str | Path) -> None:
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        self.model = NumpyPolicy(folder / "policy.npz")
        self.spec = Spec(meta["task"], meta["sensors"], meta["control_hz"], meta["max_speed"], n_rays=meta.get("n_rays", 9))
        self.policy_name = folder.name
        self.status = f"loaded {folder.name}"
        log.info("learned policy: %s (%s, %s sensors)", folder.name, self.spec.task, self.spec.sensors)

    def deployable(self) -> str | None:
        """Why this policy can't drive the current target, or None if it can."""
        if self.spec is None:
            return "no policy loaded"
        if self.link.target != "sim" and (self.spec.task == "balance" or self.spec.sensors == "rays"):
            return "this policy is sim-only (needs 50 Hz control or sim depth rays)"
        if self.spec.task == "goto" and self.goal is None:
            return "set a goal first"
        return None

    # ── Policy API ─────────────────────────────────────────────

    async def start(self) -> None:
        self.last_action = np.zeros(2, np.float32)
        if self.spec and self.spec.task == "balance" and self.link.target == "sim" and self.link.sim:
            self.link.sim.controller = self._sim_controller
            self.link.sim.controller_hz = self.spec.control_hz

    async def stop(self) -> None:
        if self.link.sim is not None:
            self.link.sim.controller = None
        await self.body.stop()

    async def step(self, obs: Obs) -> None:
        why = self.deployable()
        if why or self.spec.task == "balance":
            self.status = why or "balancing (in the physics loop)"
            if why:
                await self.body.drive(0, 0, "learned")
            return
        s = self._snapshot(obs)
        if self.spec.task == "goto" and math.dist(s["pos"][:2], self.goal) < 0.06:
            self.status = "arrived"
            await self.body.drive(0, 0, "learned")
            return
        rays = self.link.sim.rays(self.spec.n_rays) if self.spec.sensors == "rays" and self.link.sim else None
        o = observe(self.spec, s, self.goal, rays, self.last_action)
        a, _ = self.model.predict(o, deterministic=True)
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        self.last_action = a
        self.status = "driving"
        await self.body.apply(Action(float(a[0]) * self.spec.max_speed, float(a[1]) * self.spec.max_speed,
                                     self.body.cmd.head_rate, self.body.cmd.lift_rate), source="learned")

    def _snapshot(self, obs: Obs) -> dict:
        """The same fields the sim snapshot has, from World (works on the real robot)."""
        w = obs.world
        if self.link.target == "sim" and self.link.sim and self.link.sim.snapshot:
            return self.link.sim.snapshot
        x, y, yaw = w.pose or (0.0, 0.0, 0.0)
        ax, ay, az = w.accel or (0.0, 0.0, 9800.0)
        g = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        return {"pos": [x / 1000, y / 1000, 0.0], "yaw": yaw, "left_mmps": w.left_mmps or 0.0,
                "right_mmps": w.right_mmps or 0.0, "prox_m": (w.proximity_mm / 1000) if w.proximity_mm else None,
                "cliff": w.cliff, "upright": az / g, "pitch": math.asin(max(-1, min(1, ax / g))),
                "gyro": list(w.gyro or (0.0, 0.0, 0.0))}

    def _sim_controller(self, s: dict) -> tuple[float, float]:
        """Called from the physics thread at control_hz (balance)."""
        o = observe(self.spec, s, None, None, self.last_action)
        a, _ = self.model.predict(o, deterministic=True)
        v = float(np.clip(a[0], -1, 1))
        self.last_action = np.array([v, v], np.float32)
        return v * self.spec.max_speed, v * self.spec.max_speed
