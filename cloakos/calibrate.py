"""Calibration: measure how the real pipeline differs from the training sim, as a small JSON.

  map    (simulator + camera AI) compare the AI-built map's range readings with the sim's true ones
  robot  (real Vector or sim)    command lag, stop lag, wheel speed ratio and distance-sensor noise

The numbers go into data/calibration.json; paste them to Claude to tune training to the real thing.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .rl.envs import RAY_FOV, RAY_MAX

if TYPE_CHECKING:
    from .app import Brain


class Calibrator:
    def __init__(self, brain: "Brain", path: str | Path = "data/calibration.json") -> None:
        self.brain, self.path = brain, Path(path)
        self.task: asyncio.Task | None = None
        self.kind = ""
        self.progress = ""
        self.result: dict[str, Any] = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, kind: str, seconds: float = 180.0) -> str | None:
        """None when started, otherwise why not."""
        b = self.brain
        if self.running:
            return "A calibration is already running."
        if not b.link.connected:
            return "Connect Vector (or start the simulator) first."
        if kind == "map" and b.link.target != "sim":
            return "The map comparison needs the simulator (it has the true answers)."
        if kind not in ("map", "robot"):
            return "Unknown calibration."
        self.kind, self.progress = kind, "starting"
        self.task = asyncio.ensure_future(self._run(kind, seconds))
        return None

    async def _run(self, kind: str, seconds: float) -> None:
        try:
            out = await (self._map(seconds) if kind == "map" else self._robot())
            self.result = {**self.result, kind: out, "target": self.brain.link.target, "when": time.time()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.result, indent=1))
            self.progress = "done"
            from . import share
            if share.enabled():
                try:
                    await asyncio.to_thread(share.upload_calibration, {kind: out}, self.brain.link.target)
                    self.progress = "done · shared with Claude"
                except Exception as e:
                    self.progress = f"done (couldn't share: {e})"[:200]
        except asyncio.CancelledError:
            self.progress = "stopped"
            raise
        except Exception as e:
            self.progress = f"failed: {type(e).__name__}: {e}"

    # ── map vs truth ───────────────────────────────────────────

    async def _map(self, seconds: float) -> dict:
        b = self.brain
        sim, mp = b.link.sim, b.mapper
        n = 11
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            sim.touch()
            self.progress = f"sampling {len(pairs)} readings, {int(end - time.monotonic())} s left (drive around or use Auto mode)"
            if mp.map is not None and sim.snapshot:
                s = sim.snapshot
                got = await asyncio.to_thread(mp.rays, s["pos"], s["yaw"], n, RAY_FOV, RAY_MAX)
                pairs.append((np.asarray(got), sim.rays(n, RAY_FOV, RAY_MAX)))
            await asyncio.sleep(0.5)
        if len(pairs) < 20:
            raise RuntimeError("not enough map readings (is the map running?)")
        got = np.stack([p[0] for p in pairs])
        true = np.stack([p[1] for p in pairs])
        seen = got < RAY_MAX - 1e-3
        real = true < RAY_MAX - 1e-3
        both = seen & real
        err = (got - true)[both]
        return {
            "samples": int(got.size), "mode": mp.status,
            "missed_fraction": round(float((real & ~seen).sum() / max(1, real.sum())), 3),     # real thing, map saw nothing
            "phantom_fraction": round(float((seen & ~real).sum() / max(1, seen.sum())), 3),    # map hit, nothing there
            "range_bias_m": round(float(err.mean()), 4) if err.size else None,
            "range_noise_m": round(float(err.std()), 4) if err.size else None,
            "range_noise_rel": round(float(np.median(np.abs(err) / np.maximum(true[both], 0.05))), 3) if err.size else None,
        }

    # ── command lag and sensor noise ───────────────────────────

    async def _robot(self) -> dict:
        b = self.brain
        w, body = b.world, b.body
        if b.frozen:
            raise RuntimeError("Vector is frozen: unfreeze it first.")
        prev_mode = b.mode
        await b.set_mode("idle")
        lags, stops, ratios = [], [], []
        try:
            for i in range(5):
                self.progress = f"trial {i + 1} of 5 (Vector moves about 10 cm each way)"
                sign = 1 if i % 2 == 0 else -1
                t0 = time.monotonic()
                await body.drive(100.0 * sign, 100.0 * sign, "calibrate")
                lag = None
                top = []
                while time.monotonic() - t0 < 0.9:
                    v = (w.left_mmps or 0.0) * sign
                    if lag is None and v > 50:
                        lag = time.monotonic() - t0
                    if lag is not None:
                        top.append(v)
                    await asyncio.sleep(0.02)
                t1 = time.monotonic()
                await body.drive(0, 0, "calibrate")
                stop = None
                while time.monotonic() - t1 < 1.0:
                    if abs(w.left_mmps or 0.0) < 10:
                        stop = time.monotonic() - t1
                        break
                    await asyncio.sleep(0.02)
                if lag is not None:
                    lags.append(lag)
                if stop is not None:
                    stops.append(stop)
                if top:
                    ratios.append(float(np.median(top)) / 100.0)
                await asyncio.sleep(0.4)
            prox = []
            for _ in range(40):
                if w.proximity_mm:
                    prox.append(w.proximity_mm)
                await asyncio.sleep(0.05)
        finally:
            await body.stop()
            await b.set_mode(prev_mode)
        med = lambda a: round(float(np.median(a)), 3) if a else None
        return {
            "command_to_motion_s": med(lags), "stop_s": med(stops), "wheel_speed_ratio": med(ratios),
            "trials_ok": len(lags),
            "distance_sensor": {"readings": len(prox), "std_mm": round(float(np.std(prox)), 2) if len(prox) > 5 else None},
        }
