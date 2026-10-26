"""Phase 1 behaviors. Scores are simple drive formulas; tune weights in config.yaml.

Expression goes through body.emote(): Cloak's own eyes + motion + chirp clips
(expression/emotes.py), optionally blended with Vector's stock animations.
"""
from __future__ import annotations

import random

from ..connection import Body
from ..world import World
from .base import Behavior, blocked, held, sleep


class Wander(Behavior):
    name = "wander"
    min_s = 8.0

    def score(self, d, w):
        return 0.45 * d["curiosity"] + 0.45 * d["energy"] - 0.25 * d["boredom"]

    async def run(self, body: Body, w: World, d: dict):
        await body.head_angle(5)
        steer = 0.0
        while True:
            if held(w):
                await body.drive(0, 0, self.name)
            elif blocked(w):
                await body.drive(0, 0, self.name)
                await body.straight(-50, 80)
                await body.turn(random.choice((-1, 1)) * random.uniform(70, 150))
            else:
                speed = 40 + 60 * random.random()
                steer = max(-1.0, min(1.0, steer + random.uniform(-0.25, 0.25)))
                await body.drive(speed * (1 + 0.4 * steer), speed * (1 - 0.4 * steer), self.name)
                if random.random() < 0.01:
                    await body.drive(0, 0, self.name)
                    await body.emote("curious")
            await sleep(0.1)


class LookAround(Behavior):
    name = "look_around"
    min_s = 6.0

    def score(self, d, w):
        return 0.15 + 0.5 * d["curiosity"] + 0.2 * (1 - d["energy"])

    async def run(self, body: Body, w: World, d: dict):
        while True:
            await body.head_angle(random.uniform(-5, 40))
            if not held(w):
                await body.turn(random.uniform(-70, 70), 60)
            await sleep(random.uniform(1.0, 2.5))
            if random.random() < 0.25:
                await body.emote("curious")


class ApproachFace(Behavior):
    name = "approach_face"
    min_s = 3.0

    def score(self, d, w):
        return (0.55 + 0.6 * d["attachment"] - 0.4 * d["fear"]) if w.face_visible(1.5) else 0.0

    async def run(self, body: Body, w: World, d: dict):
        await body.head_angle(30)
        greeted = False
        lost_for = 0.0
        while lost_for < 3.0:
            if not w.face_visible(0.6) or held(w):
                lost_for += 0.1
                await body.drive(0, 0, self.name)
            else:
                lost_for = 0.0
                f = w.face
                turn = (f.cx - 0.5) * 2              # -1 left .. +1 right
                close = f.size > 0.22 or blocked(w, 90)
                fwd = 0 if close else 60
                await body.drive(fwd + 70 * turn, fwd - 70 * turn, self.name)
                if close and not greeted:
                    greeted = True
                    await body.drive(0, 0, self.name)
                    await body.emote("greet")
            await sleep(0.1)


class Sulk(Behavior):
    name = "sulk"
    min_s = 12.0

    def score(self, d, w):
        return 1.2 * d["boredom"] * (1 - d["attachment"])

    async def run(self, body: Body, w: World, d: dict):
        await body.emote("sad")
        if not held(w):
            await body.turn(random.choice((-1, 1)) * 140, 50)
        await body.head_angle(-15)
        while True:
            await sleep(random.uniform(5, 9))
            if random.random() < 0.5:
                await body.emote(random.choice(("sad", "bored")))
                await body.head_angle(-15)


class Nap(Behavior):
    name = "nap"
    min_s = 30.0
    resting = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.wake_at = float(self.cfg.get("wake_at", 0.85))

    def score(self, d, w):
        e = d["energy"]
        return 1.3 * (1 - e) if e < 0.35 else 0.4 * (1 - e)

    async def run(self, body: Body, w: World, d: dict):
        await body.stop()
        await body.emote("sleepy")
        await body.head_angle(-22, 2)
        while d["energy"] < self.wake_at:
            await sleep(4)               # the eyes stay shut while napping (expression/eyes.py)

    async def exit(self, body: Body):
        await body.stop()
        await body.emote("wake")


class Startle(Behavior):
    """Reflex: fear spikes cut in immediately."""
    name = "startle"
    min_s = 2.0
    interrupt = True

    def score(self, d, w):
        return 1.6 * d["fear"] if d["fear"] > 0.6 else 0.0

    async def run(self, body: Body, w: World, d: dict):
        await body.stop()
        await body.emote("startle")
        if not held(w):
            await body.emote("scared")
        await sleep(1.5)


class Idle(Behavior):
    """Fallback when nothing scores well: sit, breathe, glance."""
    name = "idle"
    min_s = 4.0

    def score(self, d, w):
        return 0.2

    async def run(self, body: Body, w: World, d: dict):
        while True:
            await sleep(random.uniform(3, 7))
            await body.head_angle(random.uniform(0, 25))


ALL = (Startle, ApproachFace, Nap, Sulk, Wander, LookAround, Idle)
