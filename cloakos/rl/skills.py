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

from .envs import Spec, make_spec, observe
from .policy import NumpyPolicy, ranges, robot_snapshot

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
        self.reload()

    def reload(self) -> None:
        """(Re)load the shipped policies: a downloaded Nous update if there is one, else the packaged set."""
        from ..nous import active_dir
        root = active_dir()
        self.skills = {}
        for d in sorted(root.glob("*")) if root.exists() else []:
            if d.is_dir() and (d / "policy.npz").exists():
                try:
                    s = Skill(d)
                except Exception as e:
                    log.warning("skill %s didn't load: %s", d.name, e)
                    continue
                if not (s.exam or {}).get("passed"):           # only skills that passed their exam drive the robot
                    log.info("skill %s skipped: it hasn't passed its exam yet", d.name)
                    continue
                self.skills[s.spec.task] = s
        if self.skills:
            log.info("trained skills: %s", ", ".join(self.skills))

    def has(self, task: str) -> bool:
        return task in self.skills

    # ── what the brain sees ────────────────────────────────────

    def snapshot(self) -> dict:
        """The sensor dict the skills were trained on, from the sim or the real robot."""
        return robot_snapshot(self.link, self.world)

    def act(self, task: str, goal: tuple[float, float] | None = None) -> tuple[float, float, dict]:
        """One 10 Hz decision: wheel speeds (mm/s) plus a little info for the UI. Exactly the
        training loop's path (the edge reflex runs robot-side, in RobotLink, like in training)."""
        sk = self.skills[task]
        s = self.snapshot()
        r = ranges(self.link, self.mapper, s, sk.spec.n_rays) if sk.spec.sensors == "map" else None
        a, _ = sk.model.predict(observe(sk.spec, s, goal, r, sk.last_action))
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        sk.last_action = a[:2]
        info: dict[str, Any] = {}
        if goal is not None:
            info["dist"] = math.dist(s["pos"][:2], goal)
        return float(a[0]) * sk.spec.max_speed, float(a[1]) * sk.spec.max_speed, info

    def summary(self) -> dict:
        return {t: {"name": s.name, "exam": {k: v for k, v in (s.exam or {}).items() if k not in ("runs",)}}
                for t, s in self.skills.items()}
