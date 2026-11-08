"""Senses: turn raw robot state into discrete `sense.*` events (edges, debounced).

This is the slot for the vision model later: a YOLO detector publishes the same
raw.face / raw.object events and nothing downstream changes.
"""
from __future__ import annotations

import time

from ..bus import Bus, Event
from ..world import World


class Senses:
    def __init__(self, cfg: dict, bus: Bus, world: World) -> None:
        c = cfg.get("senses", {}) or {}
        self.face_forget_s = float(c.get("face_forget_s", 6))
        self.touch_debounce_s = float(c.get("touch_debounce_s", 1.5))
        self.sound_cooldown_s = float(c.get("sound_cooldown_s", 3))
        self.bus, self.world = bus, world
        self._prev = {"picked_up": False, "cliff": False, "touching": False, "button": False}
        self._last: dict[str, float] = {}
        self._face_present = False
        bus.on("raw.face", self._on_face)
        bus.on("raw.wake_word", lambda ev: self._fire("heard_wake_word"))
        bus.on("raw.loud_sound", self._on_sound)

    def _fire(self, name: str, debounce: float = 0.0, **data) -> None:
        now = time.time()
        if now - self._last.get(name, 0.0) < debounce:
            return
        self._last[name] = now
        self.bus.emit(f"sense.{name}", **data)

    def update(self) -> None:
        """Call once per tick, after the link has polled fresh state."""
        w, p = self.world, self._prev
        if w.picked_up and not p["picked_up"]:
            self._fire("picked_up")
        elif p["picked_up"] and not w.picked_up:
            self._fire("put_down")
        if w.cliff and not p["cliff"] and not w.picked_up:
            self._fire("cliff", debounce=2.0)
        if w.touching and not p["touching"]:
            self._fire("touched", debounce=self.touch_debounce_s)
        if w.button and not p["button"]:
            self._fire("button")
        p.update(picked_up=w.picked_up, cliff=w.cliff, touching=w.touching, button=w.button)

        if self._face_present and not w.face_visible(self.face_forget_s):
            self._face_present = False
            self._fire("face_lost")

    def _on_face(self, ev: Event) -> None:
        self.world.face = ev.data["face"]
        if not self._face_present:
            self._face_present = True
            self._fire("face_seen", face_id=self.world.face.id)

    def _on_sound(self, ev: Event) -> None:
        self._fire("loud_sound", debounce=self.sound_cooldown_s, level=ev.data.get("level"))
