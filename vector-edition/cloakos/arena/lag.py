"""How late the overhead camera is: compare its heading with Vector's own odometry while turning.

The odometry heading arrives within a few tens of ms; the camera's heading of the same turn shows
up later (phone capture, Continuity, the page's processing, the POST). Try candidate delays and
keep the one where the camera's turn rate best matches the odometry's, shifted by that delay.
"""
from __future__ import annotations

import math
from collections import deque

LAGS = [i * 0.02 for i in range(0, 41)]          # 0 .. 0.8 s


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class LagEstimator:
    def __init__(self, keep_s: float = 8.0) -> None:
        self.odo: deque[tuple[float, float]] = deque()
        self.cam: deque[tuple[float, float]] = deque()
        self.keep_s = keep_s
        self.lag: float | None = None

    def add_odo(self, t: float, yaw: float) -> None:
        self.odo.append((t, yaw)); self._trim(t)

    def add_cam(self, t: float, yaw: float) -> None:
        self.cam.append((t, yaw)); self._trim(t)

    def _trim(self, t: float) -> None:
        for q in (self.odo, self.cam):
            while q and t - q[0][0] > self.keep_s:
                q.popleft()

    def _odo_at(self, t: float) -> float | None:
        o = self.odo
        if not o or t < o[0][0] or t > o[-1][0]:
            return None
        for (t0, y0), (t1, y1) in zip(o, list(o)[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return y0 + f * _wrap(y1 - y0)
        return o[-1][1]

    def estimate(self) -> float | None:
        """Best delay (s), or None if Vector hasn't turned enough in the window to tell."""
        cam = list(self.cam)
        if len(cam) < 10:
            return None
        turned = sum(abs(_wrap(b[1] - a[1])) for a, b in zip(cam, cam[1:]))
        if turned < 1.0:                                  # about 60° of turning needed
            return None
        best = None
        for lag in LAGS:
            err, n = 0.0, 0
            for (ta, ya), (tb, yb) in zip(cam, cam[1:]):
                oa, ob = self._odo_at(ta - lag), self._odo_at(tb - lag)
                if oa is None or ob is None:
                    continue
                d = abs(_wrap(yb - ya)) - abs(_wrap(ob - oa))       # magnitudes: works mirrored too
                err += d * d; n += 1
            if n >= 8 and (best is None or err / n < best[0]):
                best = (err / n, lag)
        if best is None:
            return None
        self.lag = best[1] if self.lag is None else 0.8 * self.lag + 0.2 * best[1]
        return self.lag
