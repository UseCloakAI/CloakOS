"""Heading gate for the overhead camera's heading (colour markers, backpack light, motion).

Those cues flip and jump (glare, the wrong blob pair) while Vector sits still. A new reading is
believed only if it's close to the last good heading, allowing for however far Vector's own
odometry says it has turned since. A jump has to be repeated by many readings in a row before we
re-lock, far more while Vector hasn't moved (a still robot can't have turned 90°). Until a lock,
the heading is 'untrusted' and ranges are refused. set() pins it (the user said which way it faces).
"""
from __future__ import annotations

import math

JUMP = 0.35          # rad: more than ~20° from the last good heading is a jump
RELOCK = 3           # consistent readings in a row to accept a jump while turning (or to start)
RELOCK_STILL = 30    # ...and while Vector hasn't turned (~3 s at 10 Hz)
ALPHA = 0.5          # smoothing toward an accepted reading when still


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class HeadingGate:
    def __init__(self) -> None:
        self.yaw: float | None = None
        self.odo: float | None = None
        self.cand: list[float] = []

    def set(self, yaw: float, odo: float | None = None) -> None:
        self.yaw, self.odo, self.cand = wrap(yaw), odo, []

    def update(self, yaw: float, src: str, odo: float | None = None) -> tuple[float | None, bool]:
        """Feed one camera heading; returns (filtered heading, trusted)."""
        turned = abs(wrap(odo - self.odo)) if odo is not None and self.odo is not None else None
        self.odo = odo if odo is not None else self.odo
        yaw = wrap(yaw)
        if self.yaw is not None:
            d = abs(wrap(yaw - self.yaw))
            if d < JUMP + (turned or 0.0):            # either way round: the camera view may be mirrored
                self.yaw = yaw if (turned or 0) > 0.02 else wrap(self.yaw + ALPHA * wrap(yaw - self.yaw))
                self.cand = []
                return self.yaw, True
        if self.cand and abs(wrap(yaw - self.cand[-1])) >= JUMP + (turned or 0.0):
            self.cand = []
        self.cand.append(yaw)
        need = RELOCK if self.yaw is None or (turned is not None and turned > 0.02) else RELOCK_STILL
        if turned is None and self.yaw is not None:
            need = RELOCK_STILL                        # no odometry: can't tell it turned, be strict
        if len(self.cand) >= need:
            s, c = sum(math.sin(a) for a in self.cand), sum(math.cos(a) for a in self.cand)
            self.yaw, self.cand = math.atan2(s, c), []
            return self.yaw, True
        return self.yaw, False
