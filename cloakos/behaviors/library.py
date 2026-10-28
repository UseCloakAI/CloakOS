"""Phase 1 behaviors. Scores are simple drive formulas; tune weights in config.yaml.

Expression goes through body.emote(): Cloak's own eyes + motion + chirp clips
(expression/emotes.py), optionally blended with Vector's stock animations.
"""
from __future__ import annotations

import math
import random
import time

from ..connection import Body
from ..connection.body import Action
from ..world import World
from .base import Behavior, blocked, held, near_edge, sleep


class Wander(Behavior):
    """Exploring, the way an animal does. No scripted moves: every tick the path comes from a few
    continuous forces, weighted by its drives:

      wobble     smooth, never-repeating noise in heading and pace (energy = livelier)
      novelty    pull toward things it sees but hasn't checked out, and toward unexplored space (curiosity)
      avoid      push away from what's close in front and from edges
      attention  interest builds near something new, boredom builds while nothing happens; either one
                 makes it stop and look, for as long as its mood says
    The head and eyes follow whatever is pulling at it.
    """
    name = "wander"
    min_s = 8.0

    def score(self, d, w):
        return 0.45 * d["curiosity"] + 0.45 * d["energy"] - 0.25 * d["boredom"]

    async def run(self, body: Body, w: World, d: dict):
        rnd = random.Random()
        phases = [rnd.uniform(0, 2 * math.pi) for _ in range(6)]
        freqs = [rnd.uniform(0.05, 0.13), rnd.uniform(0.17, 0.31), rnd.uniform(0.4, 0.7),
                 rnd.uniform(0.07, 0.15), rnd.uniform(0.2, 0.45), rnd.uniform(0.25, 0.6)]
        seen: dict[str, float] = {}               # label -> when it last investigated one
        interest = boredom = 0.0
        steer = pace = 0.0
        head_goal = 5.0
        t0 = time.time()
        dt = 0.1

        def noise(k: int, t: float) -> float:     # smooth, non-repeating: sum of incommensurate sines
            return (math.sin(freqs[k] * 2 * math.pi * t + phases[k]) * 0.6
                    + math.sin(freqs[k] * 2 * math.pi * 2.71 * t + phases[k] * 1.7) * 0.4)

        while True:
            t = time.time() - t0
            if held(w):
                await body.drive(0, 0, self.name)
                await sleep(0.3)
                continue
            energy, curiosity, bore = d.get("energy", 0.6), d.get("curiosity", 0.5), d.get("boredom", 0.2)

            # novelty: the most interesting thing in view that it hasn't looked at lately
            target = self._target(body, seen)
            pull = 0.0
            if target is not None:
                bearing, size, label = target
                pull = bearing * (0.6 + curiosity)
                interest += dt * (0.4 + curiosity) * (0.5 + size * 4)
                if interest > 1.0 and size > 0.18:        # close enough to have a proper look
                    seen[label] = time.time()
                    interest = 0.0
                    await self._look(body, w, d, rnd, toward=bearing)
                    continue
            else:
                interest = max(0.0, interest - dt * 0.5)
                boredom += dt * (0.03 + 0.1 * bore) * (1 + max(0.0, noise(5, t)))

            # avoid what's close in front and edges
            avoid = 0.0
            if w.proximity_mm is not None and w.proximity_mm < 220:
                avoid = (1.0 - w.proximity_mm / 220) * (1 if noise(4, t) >= 0 else -1) * 1.6
            if near_edge(w):
                await body.drive(0, 0, self.name)
                await body.turn(rnd.choice((-1, 1)) * rnd.uniform(90, 170), 70)
                boredom = 0.0
                continue

            # boredom makes it stop and take stock (a hazard, not a timer)
            if boredom > 1.0 and rnd.random() < dt * boredom:
                boredom = 0.0
                await self._look(body, w, d, rnd, toward=None)
                continue

            # the path: wobble + pull + avoid, eased so it never jerks
            wobble = noise(0, t) * 0.32 + noise(1, t) * 0.14 * (0.5 + energy)
            steer_goal = max(-1.0, min(1.0, wobble + pull + avoid))
            steer += (steer_goal - steer) * min(1.0, dt * 3.0)
            pace_goal = (45 + 55 * energy) * (0.75 + 0.25 * noise(3, t)) * (1.0 - 0.6 * abs(avoid))
            if target is not None:
                pace_goal *= 0.55 + 0.45 * (1 - min(1.0, target[1] * 3))     # creep up on things
            pace += (pace_goal - pace) * min(1.0, dt * 2.0)
            spin = steer * (35 + 25 * energy)
            await body.apply(Action(pace - spin, pace + spin, self._head_rate(w, head_goal), 0.0), self.name)

            # head and eyes lead toward whatever's pulling
            head_goal = 6 + 8 * noise(2, t) + (10 if target is not None else 0)
            self._gaze(body, steer, 0.15 * noise(2, t) - (0.25 if target is not None else 0.0))
            await sleep(dt)

    @staticmethod
    def _target(body: Body, seen: dict[str, float]):
        """(bearing -1..1 left+, apparent size 0..1, label) of the freshest thing in view, or None."""
        mapper = getattr(getattr(body, "skills", None), "mapper", None)
        dets = getattr(mapper, "dets", None) or []
        fw = (getattr(mapper, "det_size", (640, 360)) or (640, 360))[0]
        best = None
        for det in dets:
            label = det.get("label", "")
            if time.time() - seen.get(label, 0.0) < 45:      # looked at one of these recently
                continue
            x0, _, x1, _ = det["box"]
            size = (x1 - x0) / fw
            bearing = -(((x0 + x1) / 2) / fw - 0.5) * 2
            score = size * det.get("score", 1.0)
            if best is None or score > best[0]:
                best = (score, bearing, size, label)
        return best[1:] if best else None

    async def _look(self, body: Body, w: World, d: dict, rnd: random.Random, toward: float | None) -> None:
        """Stop and take something in. How long and how intently follows the mood."""
        await body.drive(0, 0, self.name)
        curiosity, energy = d.get("curiosity", 0.5), d.get("energy", 0.6)
        if toward is not None:
            await body.turn(toward * 40, 50 + 40 * energy)
            await body.head_angle(rnd.uniform(5, 25), 4 + 6 * energy)
            self._gaze(body, 0.0, -0.2)
            if rnd.random() < curiosity:
                await body.emote("curious")
        else:
            glances = 1 + int(rnd.random() * (1 + 3 * curiosity))
            for _ in range(glances):
                self._gaze(body, rnd.uniform(-0.8, 0.8), rnd.uniform(-0.4, 0.2))
                await body.head_angle(rnd.uniform(-5, 38), 3 + 6 * energy)
                if rnd.random() < 0.5:
                    await body.turn(rnd.uniform(-45, 45), 30 + 40 * energy)
                await sleep(rnd.uniform(0.3, 1.2) * (1.5 - energy))
        await sleep(rnd.uniform(0.4, 1.5) * (0.6 + curiosity) * (1.6 - energy))

    @staticmethod
    def _head_rate(w: World, goal_deg: float) -> float:
        err = math.radians(goal_deg) - (w.head_angle or 0.0)
        return max(-0.6, min(0.6, err * 2.5))

    @staticmethod
    def _gaze(body: Body, x: float, y: float) -> None:
        eyes = getattr(getattr(body, "emotes", None), "eyes", None)
        if eyes is not None:
            eyes.hint = (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)), time.time() + 0.6)


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
