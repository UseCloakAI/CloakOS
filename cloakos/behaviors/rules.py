"""RulePolicy: score every behavior from the drives, run the winner as a cancellable task.

Anti-flicker: the running behavior gets a stickiness bonus and can't be replaced before
its min_s, unless the challenger is a reflex (interrupt=True).
"""
from __future__ import annotations

import asyncio
import logging
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

    @property
    def current(self) -> str:
        return self.active.name if self.active else "-"

    @property
    def resting(self) -> bool:
        return bool(self.active and self.active.resting)

    async def step(self, obs: Obs) -> None:
        self.scores = {b.name: round(b.weight * b.score(obs.drives, obs.world), 3) for b in self.behaviors}
        best = max(self.behaviors, key=lambda b: self.scores[b.name])
        cur = self.active
        if cur is None or self._task is None or self._task.done():
            await self._switch(best)
            return
        if best is cur:
            return
        ran = time.time() - self._since
        beats = self.scores[best.name] > self.scores[cur.name] + self.stickiness
        if beats and (ran >= cur.min_s or (best.interrupt and not cur.interrupt)):
            await self._switch(best)

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
