"""Calibration: measure how the real pipeline differs from the training sim, as a small JSON.

  map    (simulator + camera AI) compare the AI-built map's range readings with the sim's true ones
  robot  (real Vector or sim)    command lag, stop lag, wheel speed ratio and distance-sensor noise
  camera (real Vector or sim)    how late camera frames arrive: nod the head, time when the picture moves

The numbers go into data/calibration.json; paste them to Claude to tune training to the real thing.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .connection import Action
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
        if kind not in ("map", "robot", "camera"):
            return "Unknown calibration."
        self.kind, self.progress = kind, "starting"
        self.task = asyncio.ensure_future(self._run(kind, seconds))
        return None

    async def _run(self, kind: str, seconds: float) -> None:
        try:
            out = await (self._map(seconds) if kind == "map" else self._camera() if kind == "camera" else self._robot())
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

    async def _take_control(self) -> str:
        b = self.brain
        if b.frozen:
            raise RuntimeError("Vector is frozen: unfreeze it first.")
        prev_mode = b.mode
        await b.set_mode("idle")
        await b.link.set_control(True)
        for _ in range(60):
            if b.body.ready:
                return prev_mode
            await asyncio.sleep(0.05)
        await b.set_mode(prev_mode)
        raise RuntimeError("couldn't get control of Vector (is it busy?)")

    async def _give_back(self, prev_mode: str) -> None:
        b = self.brain
        await b.body.stop()
        if prev_mode == "idle":
            await b.link.set_control(False)
        else:
            await b.set_mode(prev_mode)

    async def _camera(self) -> dict:
        """Nod the head; compare when the head sensor says it moved with when the picture changed.
        Point Vector at something with detail (not a blank wall), lights on."""
        from PIL import Image

        b = self.brain
        w, body = b.world, b.body
        if w.frame is None:
            raise RuntimeError("no camera frames coming in")

        def small(img):
            return np.asarray(img.convert("L").resize((80, 45), Image.BILINEAR), np.float32)

        prev_mode = await self._take_control()
        sense, image, motion, fps = [], [], [], []
        try:
            for i in range(6):
                self.progress = f"nod {i + 1} of 6 (keep things still in front of Vector)"
                await body.apply(Action(0, 0, -1.5 if i % 2 == 0 else 1.5, 0), "calibrate")   # settle at one end
                await asyncio.sleep(1.2)
                await body.apply(Action(0, 0, 0, 0), "calibrate")
                await asyncio.sleep(1.0)
                base, base_head = small(w.frame), w.head_angle
                noise, fid, n0, t_f = [], w.frame_id, 0, time.monotonic()
                while time.monotonic() - t_f < 0.6:            # how much a still picture flickers
                    if w.frame_id != fid:
                        fid, n0 = w.frame_id, n0 + 1
                        noise.append(float(np.abs(small(w.frame) - base).mean()))
                    await asyncio.sleep(0.01)
                fps.append(n0 / 0.6)
                thresh = max(4.0, 4 * (float(np.median(noise)) if noise else 1.0))
                t0 = time.monotonic()
                await body.apply(Action(0, 0, 2.5 if i % 2 == 0 else -2.5, 0), "calibrate")
                t_head = t_img = None
                while time.monotonic() - t0 < 2.0 and (t_head is None or t_img is None):
                    if t_head is None and abs(w.head_angle - base_head) > 0.03:
                        t_head = time.monotonic() - t0
                    if t_img is None and w.frame_id != fid:
                        fid = w.frame_id
                        if float(np.abs(small(w.frame) - base).mean()) > thresh:
                            t_img = time.monotonic() - t0
                    await asyncio.sleep(0.005)
                await body.apply(Action(0, 0, 0, 0), "calibrate")
                if t_head is not None and t_img is not None:
                    sense.append(t_head)
                    image.append(t_img)
                    motion.append(t_img - t_head)
                await asyncio.sleep(0.3)
        finally:
            await self._give_back(prev_mode)
        if not image:
            raise RuntimeError("the picture never changed: point Vector at something with detail, lights on")
        med = lambda a: round(float(np.median(a)), 3) if a else None
        return {"command_to_head_sensor_s": med(sense), "command_to_picture_s": med(image),
                "picture_behind_sensors_s": med(motion), "frames_per_s": round(float(np.median(fps)), 1),
                "trials_ok": len(image)}

    async def _robot(self) -> dict:
        b = self.brain
        w, body = b.world, b.body
        if b.frozen:
            raise RuntimeError("Vector is frozen: unfreeze it first.")
        prev_mode = b.mode
        await b.set_mode("idle")                          # nothing else drives it during the test...
        await b.link.set_control(True)                    # ...but idle hands control back to Vector: take it
        for _ in range(60):
            if b.body.ready:
                break
            await asyncio.sleep(0.05)
        if not b.body.ready:
            await b.set_mode(prev_mode)
            raise RuntimeError("couldn't get control of Vector (is it on the charger or busy?)")
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
            if prev_mode == "idle":
                await b.link.set_control(False)
            else:
                await b.set_mode(prev_mode)
        if not lags:
            raise RuntimeError("Vector never moved: check it's on the floor, off the charger, and not frozen")
        med = lambda a: round(float(np.median(a)), 3) if a else None
        return {
            "command_to_motion_s": med(lags), "stop_s": med(stops), "wheel_speed_ratio": med(ratios),
            "trials_ok": len(lags),
            "distance_sensor": {"readings": len(prox), "std_mm": round(float(np.std(prox)), 2) if len(prox) > 5 else None},
        }
