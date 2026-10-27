"""Skills: the trained driving brains that ship with CloakOS, ready to use.

  wander  roam around objects without bumping (Auto mode's Wander behavior uses it)
  goto    drive to a point around obstacles ("go to the mug")

Both see the same thing in the sim and on the real robot: odometry, wheel speeds, the
distance sensor, cliff, and 11 range readings around the robot read from the live 3D map.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .envs import RAY_FOV, RAY_MAX, Spec, make_spec, observe
from .policy import NumpyPolicy

if TYPE_CHECKING:
    from ..connection import RobotLink
    from ..mapping.mapper import Mapper
    from ..world import World

log = logging.getLogger(__name__)
PRETRAINED = Path(__file__).parent / "pretrained"


class Skill:
    def __init__(self, folder: Path) -> None:
        meta = json.loads((folder / "meta.json").read_text())
        self.name = folder.name
        self.spec: Spec = make_spec(meta["task"], meta["sensors"])
        self.model = NumpyPolicy(folder / "policy.npz")
        self.exam = json.loads((folder / "exam.json").read_text()) if (folder / "exam.json").exists() else None
        self.last_action = np.zeros(2, np.float32)

    def reset(self) -> None:
        self.last_action = np.zeros(2, np.float32)


class Skills:
    def __init__(self, link: "RobotLink", world: "World", mapper: "Mapper | None" = None) -> None:
        self.link, self.world, self.mapper = link, world, mapper
        self.skills: dict[str, Skill] = {}
        for d in sorted(PRETRAINED.glob("*")) if PRETRAINED.exists() else []:
            if (d / "policy.npz").exists():
                try:
                    s = Skill(d)
                    self.skills[s.spec.task] = s
                except Exception as e:
                    log.warning("skill %s didn't load: %s", d.name, e)
        if self.skills:
            log.info("trained skills: %s", ", ".join(self.skills))

    def has(self, task: str) -> bool:
        return task in self.skills

    # ── what the brain sees ────────────────────────────────────

    def snapshot(self) -> dict:
        """The sensor dict the skills were trained on, from the sim or the real robot."""
        sim = self.link.sim
        if self.link.target == "sim" and sim is not None and sim.snapshot:
            return sim.snapshot
        w = self.world
        x, y, yaw = w.pose or (0.0, 0.0, 0.0)
        ax, ay, az = w.accel or (0.0, 0.0, 9800.0)
        g = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        return {"pos": [x / 1000, y / 1000, 0.0], "yaw": yaw, "left_mmps": w.left_mmps or 0.0,
                "right_mmps": w.right_mmps or 0.0, "prox_m": (w.proximity_mm / 1000) if w.proximity_mm else None,
                "cliff": w.cliff, "upright": az / g if g > 1 else 1.0,
                "pitch": math.asin(max(-1, min(1, ax / g))) if g > 1 else 0.0, "gyro": list(w.gyro or (0.0, 0.0, 0.0))}

    def rays(self, s: dict, n: int) -> np.ndarray | None:
        """Ranges around the robot from the live map (the real robot's only source of them)."""
        if self.mapper is not None and self.mapper.map is not None:
            return self.mapper.rays(s["pos"], s["yaw"], n, RAY_FOV, RAY_MAX)
        sim = self.link.sim
        if self.link.target == "sim" and sim is not None:
            return sim.rays(n, RAY_FOV, RAY_MAX)
        return None

    def act(self, task: str, goal: tuple[float, float] | None = None) -> tuple[float, float, dict]:
        """One 10 Hz decision: wheel speeds (mm/s) plus a little info for the UI."""
        sk = self.skills[task]
        s = self.snapshot()
        r = self.rays(s, sk.spec.n_rays) if sk.spec.sensors == "map" else None
        o = observe(sk.spec, s, goal, r, sk.last_action)
        a, _ = sk.model.predict(o)
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        sk.last_action = a[:2]
        info: dict[str, Any] = {}
        if goal is not None:
            info["dist"] = math.dist(s["pos"][:2], goal)
        return float(a[0]) * sk.spec.max_speed, float(a[1]) * sk.spec.max_speed, info

    def summary(self) -> dict:
        return {t: {"name": s.name, "exam": {k: v for k, v in (s.exam or {}).items() if k not in ("runs",)}}
                for t, s in self.skills.items()}
