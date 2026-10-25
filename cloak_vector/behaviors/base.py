"""Behavior + Policy interfaces.

A Policy decides what the body does each tick. Phase 1 has two: RulePolicy (drive-scored
behaviors) and TeleopPolicy (you). A learned ACT/diffusion policy later implements the
same three methods and gets the same Obs, so swapping it in is one line in app.py.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..connection import Body
from ..world import World


@dataclass
class Obs:
    t: float
    drives: dict[str, float]
    world: World


class Policy(ABC):
    name = "policy"
    current = "-"        # shown in the UI / logged as `behavior`
    resting = False      # feeds the drive model (energy recovers)

    async def start(self) -> None: ...

    @abstractmethod
    async def step(self, obs: Obs) -> None:
        """Called at ~10Hz. Must return fast; long-running motion goes in tasks."""

    async def stop(self) -> None: ...


class Behavior(ABC):
    """A cancellable routine. score() says how much the creature wants it right now."""

    name = "behavior"
    min_s = 4.0          # don't switch away before this, unless something interrupts
    interrupt = False    # reflexes: may cut in before the current behavior's min_s
    resting = False
    default_anims: tuple[str, ...] = ()

    def __init__(self, cfg: dict) -> None:
        self.cfg = (cfg.get("behaviors", {}) or {}).get(self.name, {}) or {}
        self.anims = list(self.cfg.get("anims", self.default_anims))
        self.weight = float(self.cfg.get("weight", 1.0))
        self.min_s = float(self.cfg.get("min_s", self.min_s))

    @abstractmethod
    def score(self, d: dict[str, float], w: World) -> float: ...

    @abstractmethod
    async def run(self, body: Body, w: World, d: dict[str, float]) -> None:
        """Loop until done (return) or cancelled. `d` is the live drive dict."""

    async def exit(self, body: Body) -> None:
        """Cleanup after cancel or finish (e.g. wake-up anim). Default: stop motors."""
        await body.stop()


async def sleep(s: float) -> None:
    await asyncio.sleep(s)


def blocked(w: World, near_mm: float = 110) -> bool:
    return w.cliff or (w.proximity_mm is not None and w.proximity_mm < near_mm)


def held(w: World) -> bool:
    return w.picked_up or w.held

