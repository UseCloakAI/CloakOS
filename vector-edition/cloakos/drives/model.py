"""Drive model. Pure math, no robot, no clock: the caller passes dt.

Each drive is a float in [0, 1] that:
  - relaxes toward its baseline:  v += (baseline - v) * (1 - exp(-dt / tau))
  - drifts with context:          boredom/curiosity creep up when idle, energy drains moving, recovers resting
  - jumps on events:              impulse table in config (face_seen -> attachment +0.15, ...)
Threshold crossings use hysteresis so a drive hovering at the line fires once, not every tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DRIVES = ("energy", "curiosity", "boredom", "attachment", "fear")


@dataclass
class DriveSpec:
    init: float = 0.5
    baseline: float = 0.5
    tau: float = 60.0           # seconds to relax ~63% of the way back to baseline
    high: float | None = None   # emits "<name>.high" when crossed upward
    low: float | None = None    # emits "<name>.low" when crossed downward
    hysteresis: float = 0.1


@dataclass
class Context:
    """What the body is doing this tick; drives the background drift."""
    moving: bool = False
    resting: bool = False
    idle_s: float = 0.0          # seconds since anything stimulating happened
    battery_low: bool = False


def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


class DriveModel:
    def __init__(self, cfg: dict) -> None:
        self.specs = {n: DriveSpec(**(cfg.get("drives", {}).get(n) or {})) for n in DRIVES}
        self.impulses: dict[str, dict[str, float]] = cfg.get("impulses", {}) or {}
        self.rates: dict[str, float] = {
            "idle_after": 10.0,
            "boredom_idle": 0.004,
            "curiosity_idle": 0.002,
            "energy_active": -0.004,     # ~a few minutes of exploring makes it want a rest
            "energy_rest": 0.007,
            "battery_low_cap": 0.3,
            **(cfg.get("rates", {}) or {}),
        }
        self.values = {n: _clamp(s.init) for n, s in self.specs.items()}
        self._armed = {(n, edge): False for n in DRIVES for edge in ("high", "low")}

    def reset(self) -> None:
        """Back to the configured starting values (in place: policies hold this dict)."""
        for n, s in self.specs.items():
            self.values[n] = _clamp(s.init)
        self._armed = dict.fromkeys(self._armed, False)

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def snapshot(self) -> dict[str, float]:
        return {n: round(v, 4) for n, v in self.values.items()}

    def update(self, dt: float, ctx: Context | None = None) -> list[str]:
        """Advance dt seconds. Returns threshold crossings like ['boredom.high']."""
        ctx = ctx or Context()
        r = self.rates
        idle = ctx.idle_s >= r["idle_after"] and not ctx.resting
        drift = {
            "boredom": r["boredom_idle"] if idle else 0.0,
            "curiosity": r["curiosity_idle"] if idle else 0.0,
            "energy": r["energy_rest"] if ctx.resting else r["energy_active"] if ctx.moving else 0.0,
        }
        for n, s in self.specs.items():
            v = self.values[n]
            v += (s.baseline - v) * (1.0 - math.exp(-dt / max(s.tau, 1e-3)))
            v += drift.get(n, 0.0) * dt
            self.values[n] = _clamp(v)
        if ctx.battery_low:
            self.values["energy"] = min(self.values["energy"], r["battery_low_cap"])
        return self._crossings()

    def impulse(self, event: str, scale: float = 1.0) -> list[str]:
        """Apply an event's deltas. Unknown events are ignored."""
        for n, delta in (self.impulses.get(event) or {}).items():
            if n in self.values:
                self.values[n] = _clamp(self.values[n] + delta * scale)
        return self._crossings()

    def set(self, name: str, value: float) -> list[str]:
        self.values[name] = _clamp(value)
        return self._crossings()

    def _crossings(self) -> list[str]:
        out = []
        for n, s in self.specs.items():
            v = self.values[n]
            if s.high is not None:
                key = (n, "high")
                if not self._armed[key] and v >= s.high:
                    self._armed[key] = True
                    out.append(f"{n}.high")
                elif self._armed[key] and v < s.high - s.hysteresis:
                    self._armed[key] = False
            if s.low is not None:
                key = (n, "low")
                if not self._armed[key] and v <= s.low:
                    self._armed[key] = True
                    out.append(f"{n}.low")
                elif self._armed[key] and v > s.low + s.hysteresis:
                    self._armed[key] = False
        return out
