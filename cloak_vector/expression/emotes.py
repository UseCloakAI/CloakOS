"""Emotes: Cloak's own expressive clips. Eyes + motion + chirp, played together.

Every emote is custom. `expression.stock_blend` (0..1) is the chance to swap in one of
Vector's built-in animations instead, matched by keyword from `expression.stock`. Set it
to 0 for a fully custom creature.
"""
from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Awaitable, Callable

from .chirps import CHIRPS, chirp
from .eyes import Eyes

if TYPE_CHECKING:
    from ..connection import Body

DEFAULT_STOCK = {
    "greet": ["GreetAfterLongTime", "Greet"], "happy": ["Happy", "Celebrat"], "curious": ["Explor", "LookAround"],
    "scared": ["Scared", "Frightened"], "startle": ["Startle", "Surprise"], "sad": ["Sad", "Frustrated"],
    "bored": ["Bored"], "sleepy": ["GoToSleepGetIn", "Sleep"], "wake": ["GoToSleepGetOut", "WakeUp"],
}


class Emotes:
    def __init__(self, cfg: dict, eyes: Eyes) -> None:
        c = cfg.get("expression", {}) or {}
        self.eyes = eyes
        self.stock_blend = float(c.get("stock_blend", 0.25))
        self.stock = {**DEFAULT_STOCK, **(c.get("stock") or {})}
        self.chirps = bool(c.get("chirps", True))
        self.chirp_volume = int(c.get("chirp_volume", 70))
        self.pitch = float(c.get("chirp_pitch", 1.0))
        self._pcm = {name: chirp(name, self.pitch) for name in CHIRPS}
        self._clips: dict[str, Callable[["Body"], Awaitable[None]]] = {
            "greet": self._greet, "happy": self._happy, "curious": self._curious, "scared": self._scared,
            "startle": self._startle, "sad": self._sad, "bored": self._bored, "sleepy": self._sleepy, "wake": self._wake,
        }

    @property
    def names(self) -> list[str]:
        return list(self._clips)

    async def play(self, name: str, body: "Body") -> None:
        if name not in self._clips:
            return
        if self.stock_blend > 0 and random.random() < self.stock_blend and await body.anim(*self.stock.get(name, [])):
            return
        self.eyes.set_mood(name, 2.5)
        jobs = [self._clips[name](body)]
        if self.chirps and name in self._pcm:
            jobs.append(body.play_pcm(self._pcm[name], self.chirp_volume))
        await asyncio.gather(*jobs)

    # ── clips: short, physical, readable from across the room ──

    async def _wiggle(self, body: "Body", n: int, speed: float, dt: float) -> None:
        for i in range(n):
            s = speed if i % 2 == 0 else -speed
            await body.drive(s, -s)
            await asyncio.sleep(dt)
        await body.drive(0, 0)

    async def _nod(self, body: "Body", lo: float, hi: float, n: int = 2) -> None:
        for _ in range(n):
            await body.head_angle(hi, 20)
            await body.head_angle(lo, 20)

    async def _greet(self, body):
        await body.head_angle(35, 15)
        await asyncio.gather(self._wiggle(body, 4, 60, 0.12), body.lift_height(0.6, 15))
        await body.lift_height(0, 15)

    async def _happy(self, body):
        await asyncio.gather(self._wiggle(body, 6, 50, 0.09), self._nod(body, 15, 30))

    async def _curious(self, body):
        await body.head_angle(25, 8)
        await body.turn(random.choice((-20, 20)), 60)
        await asyncio.sleep(0.4)

    async def _scared(self, body):
        await asyncio.gather(body.straight(-40, 150), body.head_angle(-10, 30), body.lift_height(0, 30))
        await self._wiggle(body, 6, 25, 0.05)

    async def _startle(self, body):
        await asyncio.gather(body.head_angle(40, 40), body.lift_height(0.4, 40))
        await asyncio.sleep(0.3)
        await body.head_angle(10, 10)
        await body.lift_height(0, 10)

    async def _sad(self, body):
        await body.lift_height(0, 4)
        await body.head_angle(-20, 2)
        await asyncio.sleep(1.0)

    async def _bored(self, body):
        await body.head_angle(-5, 3)
        await body.turn(random.choice((-35, 35)), 30)
        await body.head_angle(10, 3)

    async def _sleepy(self, body):
        await body.head_angle(-15, 2)
        await body.lift_height(0, 3)
        await self._nod(body, -22, -12, 1)

    async def _wake(self, body):
        await body.head_angle(20, 6)
        await self._wiggle(body, 2, 30, 0.15)
