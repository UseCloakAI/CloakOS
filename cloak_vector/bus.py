"""Tiny asyncio pub/sub. Every layer talks through topics, so nothing imports anything it doesn't need.

Topics in use:
  sense.<event>        face_seen, touched, picked_up, put_down, cliff, loud_sound, heard_wake_word
  drive.<name>.<edge>  e.g. drive.fear.high, drive.energy.low (threshold crossings)
  behavior.changed     {name}
  voice.said           {text, trigger, source}
  robot.connection     {state}
  mode.changed         {mode}
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class Event:
    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)


Handler = Callable[[Event], Any]


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def on(self, pattern: str, fn: Handler) -> None:
        """Subscribe. `pattern` is an exact topic, a prefix ending in '*' ('sense.*'), or '*'."""
        self._subs[pattern].append(fn)

    def emit(self, topic: str, **data: Any) -> Event:
        ev = Event(topic, data)
        for pattern, fns in list(self._subs.items()):
            if pattern == topic or pattern == "*" or (pattern.endswith("*") and topic.startswith(pattern[:-1])):
                for fn in fns:
                    try:
                        res = fn(ev)
                        if inspect.isawaitable(res):
                            asyncio.ensure_future(res)
                    except Exception:
                        log.exception("bus handler failed for %s", topic)
        return ev

    def emit_threadsafe(self, topic: str, **data: Any) -> None:
        """For SDK callbacks, which fire on the SDK's own thread."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(lambda: self.emit(topic, **data))
