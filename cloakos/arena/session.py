"""Arena: the real desk, watched by a camera above it (an iPhone via Continuity Camera works).

The page does the seeing: it reads the camera in the browser, finds Vector's gold top (and the
dot, for which way it faces), and maps that through the desk corners you clicked into desk
coordinates. It posts the pose here about 10 times a second.

This side keeps the desk setup and scores real runs, the same marks as the sim exam:
desk covered per minute, edge saves (times the edge reflex had to step in), falls.
A fall ends the run (someone has to put Vector back).

Charger: click it and the side Vector drives up to. Going home (low battery, the Dock button, or
anything else that sends it home) then first drives Vector to just in front of the charger, facing
it, using the camera, so Vector's own docking only has the last bit to do. After a charge it
drives off and carries on with what it was doing, including a scored run.

Desk frame: origin at the front-left corner, x to the right, y away from you, metres.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..app import Brain

CELL = 0.05
LOST_S = 3.0          # no sighting this long: tracking is lost (paused, not a fall)


class Arena:
    def __init__(self, brain: "Brain", data_dir: str | Path = "data") -> None:
        self.brain = brain
        self.file = Path(data_dir) / "arena.json"
        self.runs_file = Path(data_dir) / "arena_runs.jsonl"
        try:
            self.config: dict[str, Any] = json.loads(self.file.read_text())
        except (OSError, ValueError):
            self.config = {}
        self.pose: dict[str, Any] | None = None
        self.task: asyncio.Task | None = None
        self.run: dict[str, Any] | None = None

    # ── setup ──────────────────────────────────────────────────

    def save_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Corners (image points), colours and the desk size in cm. The size also goes to training:
        half the sim desks are then your desk's size (CLOAK_DESK)."""
        self.config.update({k: v for k, v in cfg.items() if k in ("corners", "size", "gold", "dot", "flip", "tol", "camera", "charger", "auto_dock", "mfront", "mback")})
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.config, indent=1))
        size = self.config.get("size")
        if size and all(float(v) > 0 for v in size):
            desk = f"{round(float(size[0]))}x{round(float(size[1]))}"
            os.environ["CLOAK_DESK"] = desk                   # training runs started from now on
            try:
                from ..web.setup_api import env_path, write_env
                write_env(env_path(), {"CLOAK_DESK": desk})
            except Exception:
                pass
        return self.config

    @property
    def size_m(self) -> tuple[float, float] | None:
        s = self.config.get("size")
        return (float(s[0]) / 100, float(s[1]) / 100) if s else None

    # ── ranges for "map" drivers on the real robot ─────────────

    RANGES_FRESH_S = 0.5   # older than this and Vector may have moved: no ranges (accuracy over coverage)
    CHARGER_R = 0.06       # m: the charger as a round obstacle on the desk

    def ranges_status(self) -> str | None:
        """None when the Arena can give ranges right now, otherwise why not."""
        if not self.size_m or not self.config.get("corners"):
            return "Arena desk isn't set up (corners and size)"
        p = self.pose
        if not p or not p.get("ok") or time.time() - p["t"] > self.RANGES_FRESH_S:
            return "Arena isn't tracking Vector"
        if self.heading() is None:
            return "Arena doesn't know which way Vector faces yet"
        return None

    def obstacles(self) -> list[tuple[float, float, float]]:
        """Things on the desk the Arena knows about. Only the charger today: there's no object detector yet."""
        c = self.config.get("charger")
        return [(float(c["x"]), float(c["y"]), self.CHARGER_R)] if c else []

    def ranges(self, n: int, fov: float, max_range: float):
        """Ray ranges in Vector's frame from the overhead view, like the sim's rays(); None if not live."""
        if self.ranges_status() is not None:
            return None
        from .ranges import desk_rays
        p = self.pose
        return desk_rays(float(p["x"]), float(p["y"]), float(self.heading()), self.size_m, n, fov, max_range, self.obstacles())

    def teacher_view(self, n: int = 9) -> dict[str, Any] | None:
        """What a recording keeps from the overhead camera each tick (the teacher's signals)."""
        p = self.pose
        if not p:
            return None
        from ..rl.envs import RAY_FOV, RAY_MAX
        r = self.ranges(n, RAY_FOV, RAY_MAX)
        h = self.heading()
        return {"ok": bool(p.get("ok")), "age": round(time.time() - p["t"], 2),
                "x": p.get("x"), "y": p.get("y"), "yaw": h, "yaw_src": p.get("yaw_src"),
                "ranges": None if r is None else [round(float(v), 3) for v in r], "status": self.ranges_status()}

    def update_pose(self, p: dict[str, Any]) -> None:
        self.pose = {**p, "t": time.time()}
        # the camera's heading is only sure when it saw the light or watched it move; Vector's own
        # odometry is sure for short turns (in place, where the camera can't tell). Keep the offset
        # between the two, so heading = odometry + offset is good all the time.
        odo = self._odo_yaw()
        if p.get("ok") and odo is not None and p.get("yaw_src") in ("light", "motion"):
            cam = float(p["yaw"])
            prev = self._yaw_prev
            if prev is not None:                           # do camera and Vector agree which way is left?
                dc = math.atan2(math.sin(cam - prev[0]), math.cos(cam - prev[0]))
                do = math.atan2(math.sin(odo - prev[1]), math.cos(odo - prev[1]))
                if abs(do) > 0.03 and abs(dc) < 1.0:
                    self._yaw_corr = max(-5.0, min(5.0, self._yaw_corr + dc * do))
            self._yaw_prev = (cam, odo)
            sg = self.yaw_sign
            d = math.atan2(math.sin(cam - sg * odo - self._yaw_off), math.cos(cam - sg * odo - self._yaw_off))
            self._yaw_off += d if self._yaw_fresh else 0.3 * d
            self._yaw_fresh = False

    _yaw_prev: tuple[float, float] | None = None
    _yaw_corr = 0.0

    @property
    def yaw_sign(self) -> int:
        """+1 normally; -1 if the desk view is mirrored (corners clicked the other way round)."""
        return -1 if self._yaw_corr < -0.05 else 1

    def imu(self) -> dict[str, Any]:
        """For the page: the fused heading (Vector's gyro-based heading, kept true by the camera) and
        the gyro's turn rate right now, so the arrow turns with Vector instead of a camera-lag later."""
        w = getattr(self.brain, "world", None)
        gz = float((getattr(w, "gyro", None) or (0, 0, 0))[2]) if w is not None else 0.0
        return {"yaw": self.heading() if not self._yaw_fresh else None, "om": self.yaw_sign * gz}

    _yaw_off, _yaw_fresh = 0.0, True

    def _odo_yaw(self) -> float | None:
        pose = getattr(self.brain.world, "pose", None) if hasattr(self.brain, "world") else None
        return float(pose[2]) if pose else None

    def heading(self) -> float | None:
        odo = self._odo_yaw()
        if odo is not None and not self._yaw_fresh:
            y = self.yaw_sign * odo + self._yaw_off
            return math.atan2(math.sin(y), math.cos(y))
        return float(self.pose["yaw"]) if self.pose and self.pose.get("ok") else None

    # ── the page: the right mode while you're in the Arena ──────

    _prev_mode: str | None = None

    async def enter(self) -> str:
        """Opening the Arena: Teleop, so Vector holds still for setup and you can drive it. Left alone
        if it's charging, running something of its own here, or the emergency stop is on."""
        b = self.brain
        if b.mode in ("auto", "idle", "goto") and not b.estop and not self.running and not (self.dock_task and not self.dock_task.done()):
            self._prev_mode = b.mode
            await b.set_mode("teleop")
        return b.mode

    async def leave(self) -> str:
        """Leaving the Arena: back to the mode it was in, unless a run or a trip home is still going
        (those carry on) or you've switched mode yourself."""
        b = self.brain
        if self._prev_mode and b.mode == "teleop" and not self.running and not (self.dock_task and not self.dock_task.done()):
            await b.set_mode(self._prev_mode)
        self._prev_mode = None
        return b.mode

    # ── going home ─────────────────────────────────────────────

    APPROACH = 0.18        # m in front of the charger: Vector's docking takes over from there

    def can_route(self) -> bool:
        p = self.pose
        return bool(self.config.get("charger") and self.size_m and p and p.get("ok") and time.time() - p["t"] < LOST_S)

    async def route_to_charger(self, timeout: float = 60.0) -> bool:
        """Drive to the approach point in front of the charger, then turn to face it. True if it got
        there. Wheels only (the edge reflex still guards); stops on its own if anything takes over."""
        b, c = self.brain, self.config["charger"]
        tx, ty = c["x"] + math.cos(c["yaw"]) * self.APPROACH, c["y"] + math.sin(c["yaw"]) * self.APPROACH
        face = c["yaw"] + math.pi                                    # looking at the charger
        self.dock_state = "driving to the charger"
        end, phase = time.time() + timeout, "drive"
        best_d, best_t = math.inf, time.time()            # stalled (no headway in 12 s): let Vector's own docking finish
        self.route = {"target": [tx, ty], "charger": [c["x"], c["y"]], "phase": phase}
        try:
            while time.time() < end:
                await asyncio.sleep(0.1)
                if b.estop or b.frozen or b.mode != "home":
                    return False
                p, yaw = self.pose, self.heading()
                if not p or time.time() - p["t"] > LOST_S or yaw is None:
                    await b.body.drive(0, 0, "arena")
                    continue
                x, y = float(p["x"]), float(p["y"])
                if phase == "drive":
                    d = math.hypot(tx - x, ty - y)
                    if d < best_d - 0.02:
                        best_d, best_t = d, time.time()
                    elif time.time() - best_t > 12:
                        self.dock_state = "no headway: Vector's own docking takes it from here"
                        return False
                    if d < 0.04:
                        phase = self.route["phase"] = "face"
                        continue
                    err = math.atan2(math.sin(math.atan2(ty - y, tx - x) - yaw), math.cos(math.atan2(ty - y, tx - x) - yaw))
                    # curve toward it, slower the more it's off; only spin in place when it's well off.
                    # Gentle gains: commands land ~0.4 s late over Wi-Fi, sharper ones overshoot and zigzag
                    if abs(err) > 1.2:
                        s = 45.0 if err > 0 else -45.0
                        await b.body.drive(-s, s, "arena")
                    else:
                        v = min(110.0, 30 + 350 * d) * max(0.25, math.cos(err))
                        w = max(-45.0, min(45.0, 70 * err))
                        await b.body.drive(v - w, v + w, "arena")
                else:
                    err = math.atan2(math.sin(face - yaw), math.cos(face - yaw))
                    if abs(err) < 0.15:
                        await b.body.drive(0, 0, "arena")
                        return True
                    s = (35.0 if abs(err) > 0.5 else 22.0) * (1 if err > 0 else -1)   # ease in: no overshoot
                    await b.body.drive(-s, s, "arena")
            return False
        finally:
            self.route = None
            await b.body.drive(0, 0, "arena")

    route: dict | None = None          # while driving home: where it's headed (for the page to draw)
    dock_state: str | None = None
    dock_task: asyncio.Task | None = None

    def dock(self) -> str | None:
        """Charge now, then come back: home (camera route + Vector's docking), wait for a full battery,
        drive off, and pick up the mode (and the scored run) it was in."""
        if self.dock_task and not self.dock_task.done():
            return "Already going home."
        if not self.brain.link.connected:
            return "Vector isn't connected."
        self.dock_task = asyncio.create_task(self._dock_and_return())
        return None

    async def _dock_and_return(self) -> None:
        b = self.brain
        prev = b.mode if b.mode not in ("home", "idle") else "auto"
        again = None
        if self.running:
            r = self.run
            again = (r["policy"], r["seconds"])
            self.task.cancel()
            await self._end("went to charge")
        try:
            self.dock_state = "going home"
            await b.go_home()
            if not b.world.on_charger:
                self.dock_state = "couldn't dock"
                return
            self.dock_state = "charging"
            while not (b.world.battery_level == 3 and not b.world.charging):
                await asyncio.sleep(5)
                if b.mode != "home":                                 # you took over: leave it be
                    self.dock_state = None
                    return
            self.dock_state = "charged: heading back out"
            await b.body.leave_home()
            await asyncio.sleep(1.5)
            await b.set_mode(prev)
            if again:
                await asyncio.sleep(1.0)
                await self.start(again[0], again[1])
            self.dock_state = None
        except asyncio.CancelledError:
            self.dock_state = None

    # ── scored runs ────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self, policy: str | None, seconds: float) -> str | None:
        b = self.brain
        if not self.size_m or not self.config.get("corners"):
            return "Set up the desk first (corners and size)."
        if b.estop:
            return "Release the emergency stop first."
        w = getattr(b, "world", None)
        if w is not None and w.on_charger:                   # starting from the charger: drive off it first
            if b.mode == "home":
                await b.set_mode("teleop")
            await b.body.leave_home()
            for _ in range(30):                              # until the camera sees it out on the desk
                await asyncio.sleep(0.2)
                if self.pose and self.pose.get("ok") and time.time() - self.pose["t"] < 1.0 and not w.on_charger:
                    break
        if not self.pose or time.time() - self.pose["t"] > LOST_S:
            return "The camera can't see Vector right now."
        if b.estop:
            return "Release the emergency stop first."
        if self.running:
            return "A run is already going."
        if policy:
            try:
                await asyncio.to_thread(b.learned.load, b.trainer.root / policy)
            except Exception as e:
                return f"Couldn't load {policy}: {e}"
            why = b.learned.deployable()
            if why:
                return why
            if b.mode == "learned":
                await b.set_mode("idle")
            await b.set_mode("learned")
        self.run = {"policy": policy or None, "driver": policy or f"you ({b.mode})", "target": b.link.target,
                    "start": time.time(), "seconds": float(seconds), "cells": 0, "falls": 0, "edge_saves": 0,
                    "lost_s": 0.0, "status": "running", "path": []}
        self.task = asyncio.create_task(self._watch(policy))
        return None

    async def stop(self, why: str = "stopped") -> None:
        if self.running:
            self.task.cancel()
            await self._end(why)

    async def _watch(self, policy: str | None) -> None:
        r, link = self.run, self.brain.link
        edges0 = getattr(link, "edge_events", 0)
        cells: set[tuple[int, int]] = set()
        w, d = self.size_m
        try:
            while time.time() - r["start"] < r["seconds"]:
                await asyncio.sleep(0.1)
                if policy and self.brain.mode != "learned":
                    return await self._end("mode changed")
                r["edge_saves"] = getattr(link, "edge_events", 0) - edges0
                w_ = getattr(self.brain, "world", None)
                if self.config.get("auto_dock", True) and self.config.get("charger") and w_ is not None and w_.battery_low \
                        and not w_.on_charger:
                    self.dock()                                  # tired mid-run: charge, then do the run again
                    return
                p = self.pose
                if not p or time.time() - p["t"] > LOST_S:
                    r["lost_s"] += 0.1
                    continue
                if not p.get("ok"):
                    continue
                x, y = float(p["x"]), float(p["y"])
                if x < -0.02 or y < -0.02 or x > w + 0.02 or y > d + 0.02:     # its middle is past the edge
                    r["falls"] += 1
                    return await self._end("fell")
                cells.add((math.floor(x / CELL), math.floor(y / CELL)))
                r["cells"] = len(cells)
                if not r["path"] or math.dist(r["path"][-1][:2], (x, y)) > 0.01:
                    r["path"].append((round(x, 3), round(y, 3), round(float(p.get("yaw", 0.0)), 2)))
                    del r["path"][:-3000]
        except asyncio.CancelledError:
            return
        await self._end("done")

    async def _end(self, why: str) -> None:
        r = self.run
        if r is None or r.get("status") != "running":
            return
        r["status"] = why
        r["end"] = time.time()
        if r.get("policy") and self.brain.mode == "learned":
            await self.brain.set_mode("idle")
        res = self.summary()
        try:
            with self.runs_file.open("a") as f:
                f.write(json.dumps({**res, "path": r["path"]}) + "\n")
            if r.get("policy"):                          # next to the policy, for the Training page
                pf = self.brain.trainer.root / r["policy"] / "real_exams.json"
                old = json.loads(pf.read_text()) if pf.exists() else []
                pf.write_text(json.dumps((old + [res])[-20:], indent=1))
        except (OSError, ValueError):
            pass
        self.task = None

    def summary(self) -> dict[str, Any] | None:
        r = self.run
        if r is None:
            return None
        secs = max(1e-6, (r.get("end") or time.time()) - r["start"])
        mins = secs / 60
        w, d = self.size_m or (1, 1)
        total = (w / CELL) * (d / CELL)
        return {"policy": r["policy"], "driver": r["driver"], "status": r["status"], "when": r["start"],
                "seconds": round(secs), "planned_s": r["seconds"], "cells": r["cells"],
                "covered": round(min(1.0, r["cells"] / total), 3), "cells_per_min": round(r["cells"] / mins, 1),
                "edge_saves": r["edge_saves"], "edge_saves_per_min": round(r["edge_saves"] / mins, 2),
                "falls": r["falls"], "lost_s": round(r["lost_s"], 1), "target": r["target"]}

    def recent(self, n: int = 10) -> list[dict]:
        try:
            lines = self.runs_file.read_text().splitlines()[-n:]
        except OSError:
            return []
        out = []
        for x in reversed(lines):
            try:
                out.append({k: v for k, v in json.loads(x).items() if k != "path"})
            except ValueError:
                pass
        return out
