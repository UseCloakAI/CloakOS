"""Reactive sound effects: little synthesized noises for what's happening right now (picked up,
petted, nearly fell off the table, found something, got there...). No samples.

Every sound varies a bit each time and follows the mood: more energy = higher and quicker,
fear = shaky, boredom = slower and lower. Rate-limited so it never chatters, and it stays
quiet while Vector is talking.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING

import numpy as np

from .chirps import SR, loud, note

if TYPE_CHECKING:
    from ..bus import Bus, Event
    from ..connection import Body


def _noise(dur: float, lo: float = 0.0) -> np.ndarray:
    n = int(dur * SR)
    x = np.random.default_rng().normal(0, 1, n).astype(np.float32)
    k = np.ones(8, np.float32) / 8                       # soften it
    x = np.convolve(x, k, "same")
    env = np.linspace(1, lo, n) ** 2
    return x * env


def _purr(dur: float, f: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 22 * t)        # the rumble
    x = np.sin(2 * np.pi * f * t + 0.6 * np.sin(2 * np.pi * 3 * t)) * am
    env = np.minimum(1, t / 0.15) * np.minimum(1, (dur - t) / 0.4)
    return (x * env).astype(np.float32)


def _sting(f: float, dur: float) -> np.ndarray:
    """A low, punchy brass-ish hit: square-ish wave plus an octave below, hard attack, fast decay."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.tanh(2.5 * np.sin(2 * np.pi * f * t)) + 0.6 * np.sin(np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    env = np.minimum(1, t / 0.008) * np.exp(-t / (dur * 0.45))
    return (x * env).astype(np.float32)


def _ping(f: float, dur: float) -> np.ndarray:
    """Sonar ping: pure tone, instant attack, long ringing decay with a faint echo."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.18)
    echo = int(0.22 * SR)
    x[echo:] += 0.3 * x[:-echo]
    return (x * np.minimum(1, t / 0.003)).astype(np.float32)


def _seq(parts: list[tuple[str, tuple]], vibrato: float = 0.0) -> np.ndarray:
    out = []
    for kind, a in parts:
        if kind == "n":
            x = note(*a)
            if vibrato:
                t = np.arange(len(x)) / SR
                x = x * (1 + vibrato * np.sin(2 * np.pi * 9 * t))
            out.append(x)
        elif kind == "gap":
            out.append(np.zeros(int(a[0] * SR), np.float32))
        elif kind == "noise":
            out.append(_noise(*a) * 0.5)
        elif kind == "purr":
            out.append(_purr(*a))
        elif kind == "sting":
            out.append(_sting(*a))
        elif kind == "ping":
            out.append(_ping(*a))
    return np.concatenate(out)


# name -> function(pitch p, speed s) -> list of parts. Kept short: 0.1-0.6 s, except the purr.
SOUNDS = {
    "pickup":   lambda p, s: [("n", (500 * p, 1500 * p, 0.22 * s))],                                  # whee
    "putdown":  lambda p, s: [("n", (900 * p, 600 * p, 0.08 * s)), ("gap", (0.03,)), ("n", (650 * p, 450 * p, 0.12 * s))],
    "pet":      lambda p, s: [("purr", (1.2, 320 * p))],
    "cliff":    lambda p, s: [("n", (1800 * p, 2300 * p, 0.05)), ("gap", (0.03,)), ("n", (1800 * p, 2300 * p, 0.05))],
    "button":   lambda p, s: [("n", (1300 * p, 1300 * p, 0.04))],
    "face":     lambda p, s: [("n", (700 * p, 1100 * p, 0.07 * s)), ("gap", (0.02,)), ("n", (1000 * p, 1500 * p, 0.1 * s))],
    "hm":       lambda p, s: [("n", (600 * p, 900 * p, 0.16 * s))],                                   # heard you
    "found":    lambda p, s: [("n", (1500 * p, 1500 * p, 0.05)), ("gap", (0.04,)), ("n", (2000 * p, 2000 * p, 0.07))],
    "go":       lambda p, s: [("n", (600 * p, 800 * p, 0.07 * s)), ("gap", (0.02,)), ("n", (900 * p, 1200 * p, 0.09 * s))],
    "arrived":  lambda p, s: [("n", (800 * p, 800 * p, 0.07)), ("n", (1000 * p, 1000 * p, 0.07)), ("n", (1300 * p, 1600 * p, 0.14))],
    "failed":   lambda p, s: [("n", (700 * p, 600 * p, 0.14 * s)), ("gap", (0.04,)), ("n", (560 * p, 380 * p, 0.25 * s))],
    "ack":      lambda p, s: [("n", (1000 * p, 1250 * p, 0.06)), ("gap", (0.02,)), ("n", (1250 * p, 1250 * p, 0.05))],
    "freeze":   lambda p, s: [("n", (1200 * p, 300 * p, 0.35))],                                       # power down
    "unfreeze": lambda p, s: [("n", (300 * p, 1200 * p, 0.3))],                                        # power up
    "bump":     lambda p, s: [("noise", (0.07, 0.0)), ("n", (300 * p, 220 * p, 0.08))],                # oof
    "edge":     lambda p, s: [("sting", (392.0, 0.16)), ("gap", (0.09,)), ("sting", (370.0, 0.42))],     # dun dun!
    "stuck":    lambda p, s: [("ping", (1760.0, 0.7))],                                                 # sonar ping
    "yawn":     lambda p, s: [("n", (500 * p, 750 * p, 0.3 * s)), ("n", (750 * p, 350 * p, 0.5 * s))],
    "look":     lambda p, s: [("n", (850 * p, 780 * p, 0.08 * s)), ("n", (780 * p, 1100 * p, 0.14 * s))],
}

# what triggers what, and how long before the same sound may play again (s)
EVENTS = {
    "sense.picked_up": ("pickup", 1.0), "sense.put_down": ("putdown", 1.0), "sense.touched": ("pet", 4.0),
    "sense.cliff": ("cliff", 2.0), "sense.button": ("button", 0.3), "sense.face_seen": ("face", 20.0),
    "sense.heard_wake_word": ("hm", 2.0), "map.object": ("found", 8.0), "goto.arrived": ("arrived", 1.0),
    "goto.failed": ("failed", 1.0), "command": ("ack", 1.0), "raw.edge": ("edge", 1.5), "stuck": ("stuck", 2.5),
}
BEHAVIOR_START = {"wander": ("go", 15.0), "nap": ("yawn", 60.0), "look_around": ("look", 15.0), "approach_face": ("face", 20.0)}


def render(name: str, drives: dict[str, float] | None = None, gain: float = 0.97) -> bytes:
    d = drives or {}
    energy, fear, boredom = d.get("energy", 0.6), d.get("fear", 0.0), d.get("boredom", 0.0)
    p = (0.9 + 0.25 * energy - 0.1 * boredom) * random.uniform(0.94, 1.06)
    s = (1.15 - 0.3 * energy + 0.25 * boredom) * random.uniform(0.9, 1.1)
    x = _seq(SOUNDS[name](p, s), vibrato=0.25 * max(0.0, fear - 0.3))
    x = loud(x, gain)
    return (x * 32767).astype("<i2").tobytes()


class Sfx:
    def __init__(self, cfg: dict, bus: "Bus", body: "Body", drives) -> None:
        c = cfg.get("expression", {}) or {}
        self.on = bool(c.get("sfx", True))
        self.volume = int(c.get("sfx_volume", 100))
        self.body, self.drives = body, drives
        self._last: dict[str, float] = {}
        self._any = 0.0
        for topic in EVENTS:
            bus.on(topic, self._on)
        bus.on("behavior.changed", self._on_behavior)
        bus.on("freeze.changed", lambda ev: self.play("freeze" if ev.data.get("on") else "unfreeze", 0.5))

    def _on(self, ev: "Event") -> None:
        name, cooldown = EVENTS[ev.topic]
        self.play(name, cooldown)

    def _on_behavior(self, ev: "Event") -> None:
        hit = BEHAVIOR_START.get(ev.data.get("name"))
        if hit:
            self.play(*hit)

    def play(self, name: str, cooldown: float = 1.0) -> bool:
        now = time.time()
        if not self.on or not self.body.ready or now - self._any < 0.35 or now - self._last.get(name, 0) < cooldown:
            return False
        if now < getattr(self.body, "speaking_until", 0.0):         # never talk over itself
            return False
        self._last[name] = self._any = now
        pcm = render(name, getattr(self.drives, "values", None))
        asyncio.ensure_future(self.body.play_pcm(pcm, self.volume))
        return True
