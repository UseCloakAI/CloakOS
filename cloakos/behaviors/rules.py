"""RulePolicy: score every behavior from the drives, run the winner as a cancellable task.

Anti-flicker: the running behavior gets a stickiness bonus and can't be replaced before
its min_s, unless the challenger is a reflex (interrupt=True).

Variety, without scripting: a behavior gets "tired" the longer it runs (its appeal drops)
and recovers while others run; and when it's time to change, the next one is picked among
the good options with some randomness (weighted by score), not always the single top one.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time

from ..bus import Bus
from ..connection import Body
from ..world import World
from .base import Behavior, Obs, Policy
from .library import ALL

log = logging.getLogger(__name__)


class RulePolicy(Policy):
    name = "auto"

    def __init__(self, cfg: dict, body: Body, bus: Bus, world: World, drives: dict[str, float]) -> None:
        self.body, self.bus, self.world, self.drives = body, bus, world, drives
        self.behaviors: list[Behavior] = [cls(cfg) for cls in ALL]
        self.stickiness = float((cfg.get("behaviors", {}) or {}).get("stickiness", 0.15))
        self.active: Behavior | None = None
        self.scores: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._since = 0.0
        b_cfg = cfg.get("behaviors", {}) or {}
        self.tiring = float(b_cfg.get("tiring", 0.35))           # how much appeal a fully "tired" behavior loses
        self.temperature = float(b_cfg.get("temperature", 0.07))  # randomness of the pick (0 = always the top one)
        self.fatigue: dict[str, float] = {b.name: 0.0 for b in self.behaviors}
        self._last_t = time.time()

    @property
    def current(self) -> str:
        return self.active.name if self.active else "-"

    @property
    def resting(self) -> bool:
        return bool(self.active and self.active.resting)

    async def step(self, obs: Obs) -> None:
        now = time.time()
        dt, self._last_t = min(1.0, now - self._last_t), now
        cur = self.active
        for b in self.behaviors:                          # tire while running, recover while not
            f = self.fatigue[b.name]
            if b is cur:
                f += dt / b.typical_s
            else:
                f -= dt / (b.typical_s * 1.5)
            self.fatigue[b.name] = max(0.0, min(1.5, f))
        self.scores = {b.name: round(b.weight * b.score(obs.drives, obs.world)
                                     - (0.0 if b.interrupt else self.tiring * self.fatigue[b.name]), 3)
                       for b in self.behaviors}
        best = max(self.behaviors, key=lambda b: self.scores[b.name])
        if cur is None or self._task is None or self._task.done():
            await self._switch(self._pick(cur))
            return
        if best is cur:
            return
        ran = now - self._since
        if best.interrupt and not cur.interrupt and self.scores[best.name] > self.scores[cur.name] + self.stickiness:
            await self._switch(best)                      # reflexes don't wait
            return
        if ran >= cur.min_s and self.scores[best.name] > self.scores[cur.name] + self.stickiness:
            nxt = self._pick(cur)
            if nxt is not cur:
                await self._switch(nxt)

    def _pick(self, cur: Behavior | None) -> Behavior:
        """Softmax over the non-reflex options: usually a top one, sometimes the runner-up."""
        opts = [b for b in self.behaviors if not b.interrupt or b is cur]
        top = max(self.scores[b.name] for b in opts)
        weights = [math.exp((self.scores[b.name] - top) / max(1e-3, self.temperature)) for b in opts]
        return random.choices(opts, weights)[0]

    async def _switch(self, b: Behavior) -> None:
        prev = self.active
        await self._cancel()
        if prev is not None:
            await prev.exit(self.body)
        self.active, self._since = b, time.time()
        self.body.source = b.name
        self._task = asyncio.create_task(self._run(b), name=f"behavior:{b.name}")
        self.bus.emit("behavior.changed", name=b.name, prev=prev.name if prev else None, scores=self.scores)

    async def _run(self, b: Behavior) -> None:
        try:
            await b.run(self.body, self.world, self.drives)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("behavior %s crashed", b.name)

    async def _cancel(self) -> None:
        t, self._task = self._task, None
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    async def stop(self) -> None:
        prev = self.active
        await self._cancel()
        self.active = None
        if prev is not None:
            await prev.exit(self.body)
        await self.body.stop()
