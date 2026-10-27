"""Phase 1 behaviors. Scores are simple drive formulas; tune weights in config.yaml.

Expression goes through body.emote(): Cloak's own eyes + motion + chirp clips
(expression/emotes.py), optionally blended with Vector's stock animations.
"""
from __future__ import annotations

import random
import time

from ..connection import Body
from ..world import World
from .base import Behavior, blocked, held, sleep


class Wander(Behavior):
    name = "wander"
    min_s = 8.0

    def score(self, d, w):
        return 0.45 * d["curiosity"] + 0.45 * d["energy"] - 0.25 * d["boredom"]

    async def run(self, body: Body, w: World, d: dict):
        """Like an animal exploring: pick a way, roll there smoothly for a few seconds, stop and
        look around, sometimes sit a while. Never a new random speed every tick."""
        await body.head_angle(5, 4)
        while True:
            if held(w):
                await body.drive(0, 0, self.name)
                await sleep(0.3)
                continue
            # 1. turn to a new heading, unhurried
            await body.turn(random.choice((-1, 1)) * random.uniform(25, 120), random.uniform(50, 80))
            # 2. one leg: steady speed, gently drifting curve
            await self._leg(body, w, d, random.uniform(2.0, 5.5))
            # 3. stop and take it in
            await body.drive(0, 0, self.name)
            r = random.random()
            if r < 0.35:                                   # glance up: faces are above a floor robot's eye line
                await body.head_angle(random.uniform(25, 40), 5)
                await sleep(random.uniform(0.8, 1.6))
                await body.head_angle(5, 5)
            elif r < 0.6:                                  # look side to side
                for a in (random.uniform(15, 35), -random.uniform(15, 35)):
                    await body.turn(a, 45)
                    await sleep(random.uniform(0.3, 0.7))
            rest = random.uniform(0.6, 2.0) + (random.uniform(2, 5) if d.get("energy", 1) < 0.35 else 0.0)
            await sleep(rest)

    async def _leg(self, body: Body, w: World, d: dict, seconds: float) -> None:
        speed = random.uniform(55, 85) * (0.8 + 0.4 * d.get("energy", 0.5))
        curve, curve_goal = 0.0, random.uniform(-0.25, 0.25)
        skills = getattr(body, "skills", None)
        trained = callable(getattr(skills, "has", None)) and skills.has("wander")
        end = time.time() + seconds
        while time.time() < end:
            if held(w):
                break
            if blocked(w) or w.cliff:
                await body.drive(0, 0, self.name)
                await body.straight(-40, 70)
                await body.turn(random.choice((-1, 1)) * random.uniform(80, 140), 70)
                return
            if trained:                                    # the trained brain steers around what it has mapped
                left, right, _ = skills.act("wander")
            else:
                curve += (curve_goal - curve) * 0.08       # curves ease in and out
                if random.random() < 0.03:
                    curve_goal = random.uniform(-0.3, 0.3)
                left, right = speed * (1 + curve), speed * (1 - curve)
            await body.drive(left, right, self.name)
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
                await body.turn(random.uniform(-70, 70), 50)
            await sleep(random.uniform(1.5, 3.5))           # take it in before looking elsewhere
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
