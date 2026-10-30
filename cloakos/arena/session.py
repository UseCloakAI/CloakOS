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
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..app import Brain

CELL = 0.05
SPIN_LIMIT = 2.5 * math.pi   # turned this far without getting closer: something's wrong, stop
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
        self.config.update({k: v for k, v in cfg.items() if k in ("corners", "size", "gold", "dot", "flip", "tol", "camera", "charger", "auto_dock", "mfront", "mback", "camera_lag_s", "camera_mirrored", "gyro_fusion")})
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
        if p.get("yaw_src") == "untrusted":
            return "Arena's heading is jumping around (not trusted yet)"
        return None

    def obstacles(self) -> list[tuple[float, float, float]]:
        """Things on the desk the Arena knows about. Only the charger today: there's no object detector yet."""
        c = self.config.get("charger")
        return [(float(c["x"]), float(c["y"]), self.CHARGER_R)] if c else []

    def ranges(self, n: int, fov: float, max_range: float):
        """Ray ranges in Vector's frame from the overhead view, like the sim's rays(); None if not live."""
        if self.ranges_status() is not None:
            return None
        from .ranges import buffered, desk_rays
        x, y, yaw, lw, rw = self._predicted()
        r = desk_rays(x, y, yaw, self.size_m, n, fov, max_range, self.obstacles())
        if self.yaw_sign < 0:
            r = r[::-1].copy()                         # desk-frame "left" is Vector's right in a mirrored frame
        return buffered(r, fov, (lw + rw) / 2)

    def _predicted(self) -> tuple[float, float, float, float, float]:
        """x, y, heading where Vector will be when a command sent now lands, plus wheel speeds (m/s)."""
        from .ranges import COMMAND_TO_MOTION_S, WHEEL_GAIN, predict_pose
        p = self.pose
        x, y, yaw = float(p["x"]), float(p["y"]), float(self.heading())
        # The pose is already old, and whatever the driver sends now lands ~0.4 s later: answer for
        # where Vector will be then. Wheel speeds: Vector's own, else what was asked times the gain.
        w = getattr(self.brain, "world", None)
        cmd = getattr(getattr(self.brain, "body", None), "cmd", None)
        lw, rw = getattr(w, "left_mmps", None), getattr(w, "right_mmps", None)
        if lw is None or rw is None:
            lw = (getattr(cmd, "left_mmps", 0.0) or 0.0) * WHEEL_GAIN
            rw = (getattr(cmd, "right_mmps", 0.0) or 0.0) * WHEEL_GAIN
        lw, rw = float(lw) / 1000, float(rw) / 1000
        dt = (time.time() - p["t"]) + self.camera_lag() + COMMAND_TO_MOTION_S
        x, y, yaw = self.from_phys(*predict_pose(*self.to_phys(x, y, yaw), lw, rw, dt))   # wheel maths is physical
        return x, y, yaw, lw, rw

    def snapshot(self, base: dict) -> dict | None:
        """The robot snapshot with the Arena's latency-aware desk pose, or None if not live."""
        if self.ranges_status() is not None:
            return None
        x, y, yaw, _, _ = self._predicted()
        return {**base, "pos": [x, y, 0.0], "yaw": yaw}

    def in_desk(self, x: float, y: float, margin: float = 0.06) -> bool:
        s = self.size_m
        return bool(s) and margin <= x <= s[0] - margin and margin <= y <= s[1] - margin

    def named_point(self, name: str) -> tuple[float, float] | None:
        """'left corner', 'back right corner', 'middle'...: a safe spot 10 cm in from it (desk frame)."""
        s = self.size_m
        if not s:
            return None
        n, m = name.lower(), 0.10
        if any(w in n for w in ("middle", "center", "centre")) and "corner" not in n:
            return s[0] / 2, s[1] / 2
        if "corner" not in n and not any(w in n for w in ("left", "right")):
            return None
        x = m if "left" in n else s[0] - m if "right" in n else s[0] / 2
        y = s[1] - m if any(w in n for w in ("back", "far", "top")) else m if any(w in n for w in ("front", "near", "bottom")) or "corner" in n else s[1] / 2
        return x, y

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
                "confirmed": self.heading_confirmed, "camera_lag_s": self.camera_lag(),
                "ranges": None if r is None else [round(float(v), 3) for v in r], "status": self.ranges_status()}

    def update_pose(self, p: dict[str, Any]) -> None:
        p = dict(p)
        if p.get("ok") and p.get("yaw") is not None and p.get("yaw_src") in ("markers", "light", "motion"):
            if self._gate is None:
                from .heading import HeadingGate
                self._gate = HeadingGate()
            y, trusted = self._gate.update(float(p["yaw"]), p["yaw_src"], self._odo_yaw())
            p["yaw_raw"] = p["yaw"]
            if trusted:
                p["yaw"] = y
                self._trusted = (y, self._odo_yaw())
            else:
                p["yaw_src"] = "untrusted"            # said honestly: ranges wait until it settles
        elif p.get("ok") and p.get("yaw") is not None and p.get("yaw_src") is None:
            self._trusted = (float(p["yaw"]), self._odo_yaw())   # a caller that knows the true pose (sim, tests)
            self.heading_confirmed = True
        if not p.get("ok") and self.pose and time.time() - self.pose["t"] > LOST_S:
            self.heading_confirmed = False             # lost it: the next sighting has to be confirmed again
        self.pose = {**p, "t": time.time()}
        self._note_lag(p)
        if p.get("ok") and p.get("x") is not None and p.get("yaw_src") is not None:
            self._check_motion(float(p["x"]), float(p["y"]))
        self._vote_turn_sign(p)

    # ── which way Vector's own turning goes in the desk frame: measured, corners only a prior ──
    _turn_hist = None
    _votes: list[int] = []
    sign_conflict = False          # the camera and the gyro just disagreed on turn direction

    def _vote_turn_sign(self, p: dict[str, Any]) -> None:
        """Over each ~1 s of turning, does the camera's heading change the same way as Vector's
        odometry (times the current sign)? Two votes in a row the other way: switch and save. One
        vote against: stop trusting the gyro term right away (camera-only heading) until settled."""
        odo = self._odo_yaw()
        if odo is None or not p.get("ok") or p.get("yaw_src") not in ("markers", "light"):
            return
        from collections import deque
        if self._turn_hist is None:
            self._turn_hist = deque()
        now, cam = time.monotonic(), float(p.get("yaw_raw", p["yaw"]))
        h = self._turn_hist
        h.append((now, cam, odo))
        while h and now - h[0][0] > 1.5:
            h.popleft()
        if now - h[0][0] < 0.8:
            return
        t0, c0, o0 = h[0]
        dc = 0.0                                   # camera turn, summed step by step (no wrap-around surprises)
        for (_, a, _), (_, b, _) in zip(list(h), list(h)[1:]):
            dc += math.atan2(math.sin(b - a), math.cos(b - a))
        do = math.atan2(math.sin(odo - o0), math.cos(odo - o0))
        h.clear()
        h.append((now, cam, odo))
        if abs(do) < 0.35 or abs(dc) < 0.2 or abs(abs(dc) - abs(do)) > 0.8 * abs(do) + 0.3:
            return                                 # not a clear turn (or a glitchy reading): no vote
        vote = 1 if dc * do > 0 else -1
        # vote = the sign the data implies (camera desk-yaw change vs physical odometry turn); the
        # geometry says yaw_sign. A sanity check only: 3 against in a row stops driving, never flips.
        self._votes = (self._votes + [vote])[-3:]
        if len(self._votes) == 3 and all(v != self.yaw_sign for v in self._votes):
            if not self.sign_conflict:
                log.warning("arena: camera and odometry turned opposite to the desk geometry 3 times "
                            "(last: camera %.2f, odometry %.2f rad); stopping go-to/home", dc, do)
            self.sign_conflict = True
        elif vote == self.yaw_sign:
            self.sign_conflict = False

    _yaw_prev: tuple[float, float] | None = None
    _lag = None
    _lag_n = 0

    def camera_lag(self) -> float:
        """Seconds the overhead camera runs behind (measured while turning; saved; 0.15 s until then)."""
        v = self.config.get("camera_lag_s")
        return float(v) if v is not None else 0.15

    def _note_lag(self, p: dict[str, Any]) -> None:
        odo = self._odo_yaw()
        if odo is None:
            return
        from .lag import LagEstimator
        if self._lag is None:
            self._lag = LagEstimator()
        now = time.monotonic()
        self._lag.add_odo(now, odo)
        if p.get("ok") and p.get("yaw_raw", p.get("yaw")) is not None and p.get("yaw_src") in ("markers", "light", "untrusted"):
            self._lag.add_cam(now, float(p.get("yaw_raw", p["yaw"])))
        self._lag_n += 1
        if self._lag_n % 50 == 0:                     # every ~5 s
            est = self._lag.estimate()
            if est is not None and abs(est - self.camera_lag()) > 0.02:
                self.save_config({"camera_lag_s": round(est, 3)})

    heading_confirmed = False                          # driving forward has shown which end is the front
    _motion = None

    def _forward(self) -> bool:
        """Vector is driving straight forward right now (its wheels, else what was asked)."""
        w = getattr(self.brain, "world", None)
        lw, rw = getattr(w, "left_mmps", None), getattr(w, "right_mmps", None)
        if lw is None or rw is None:
            cmd = getattr(getattr(self.brain, "body", None), "cmd", None)
            lw, rw = getattr(cmd, "left_mmps", 0.0) or 0.0, getattr(cmd, "right_mmps", 0.0) or 0.0
        lw, rw = float(lw), float(rw)
        return lw > 15 and rw > 15 and abs(lw - rw) < 0.25 * max(lw, rw)

    def _check_motion(self, x: float, y: float) -> None:
        from .heading import MotionCheck
        if self._motion is None:
            self._motion = MotionCheck()
        verdict, mdir = self._motion.update(x, y, self._forward(), self.heading())
        if verdict == "flip":                      # the camera had the ends swapped: fix it for good
            self.save_config({"flip": not self.config.get("flip")})
            self.set_heading(mdir)
            self.flips = getattr(self, "flips", 0) + 1
        if verdict in ("flip", "ok"):
            if verdict == "ok" and self._trusted is None:
                self.set_heading(mdir)
            self.heading_confirmed = True
            self.pose = {**self.pose, "confirmed": True}
    _gate = None
    _yaw_corr = 0.0

    @property
    def yaw_sign(self) -> int:
        """+1 if a physical counter-clockwise turn (seen from above; Vector's gyro/odometry +) makes the
        desk-frame yaw go UP, -1 if it makes it go DOWN. From the corner homography only (see AGENTS.md,
        "Arena frames"): the corners are clicked front-left, front-right, back-right, back-left and map
        to desk (0,0), (W,0), (W,D), (0,D), a counter-clockwise order in desk coordinates. If they are
        also counter-clockwise in image (u right, v down) coordinates, image->desk keeps orientation
        (det > 0) and, because a top-down camera image with v down is itself a mirror of the physical
        top-down view, the desk frame is a mirror too: -1. Clockwise in the image (front edge at the
        bottom, the usual setup): +1. config camera_mirrored=true (a mirrored video feed) flips it."""
        c = self.config.get("corners")
        sign = 1
        if c and len(c) == 4:
            area = sum(c[i][0] * c[(i + 1) % 4][1] - c[(i + 1) % 4][0] * c[i][1] for i in range(4))
            sign = -1 if area > 0 else 1
        return -sign if self.config.get("camera_mirrored") else sign

    @property
    def gyro_fusion(self) -> bool:
        """Carry the heading with Vector's odometry between camera readings. Off by default: the
        camera markers + heading filter + confirm-by-driving (v0.9.11) are the heading."""
        return bool(self.config.get("gyro_fusion", False))

    # physical frame: the desk frame un-mirrored, so +yaw is a physical counter-clockwise turn and
    # the wheel maths (right faster = turn left = +yaw) holds. Controllers work here.
    def to_phys(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        return (x, y, yaw) if self.yaw_sign > 0 else (x, -y, -yaw)

    def from_phys(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        return self.to_phys(x, y, yaw)                 # a mirror is its own inverse

    def imu(self) -> dict[str, Any]:
        """For the page: the fused heading (Vector's gyro-based heading, kept true by the camera) and
        the gyro's turn rate right now, so the arrow turns with Vector instead of a camera-lag later."""
        w = getattr(self.brain, "world", None)
        gz = float((getattr(w, "gyro", None) or (0, 0, 0))[2]) if w is not None else 0.0
        return {"yaw": self.heading(), "om": self.yaw_sign * gz if self.gyro_fusion else 0.0}


    def _odo_yaw(self) -> float | None:
        pose = getattr(self.brain.world, "pose", None) if hasattr(self.brain, "world") else None
        return float(pose[2]) if pose else None

    def set_heading(self, yaw: float) -> None:
        """The user says which way Vector faces right now (desk frame): pin the camera and gyro to it."""
        from .heading import HeadingGate
        if self._gate is None:
            self._gate = HeadingGate()
        odo = self._odo_yaw()
        self._gate.set(yaw, odo)
        self._trusted = (math.atan2(math.sin(yaw), math.cos(yaw)), odo)
        if self.pose:
            self.pose = {**self.pose, "yaw": yaw, "yaw_src": "set"}

    _trusted: tuple[float, float | None] | None = None     # last trusted camera heading, and odometry then

    def heading(self) -> float | None:  # noqa: C901
        """THE heading (desk frame, rad, 0 = +x, +pi/2 = +y = toward the back edge): the last trusted
        camera heading (plus Vector's odometry turn since, only with gyro_fusion on). The page shows
        this same number in its text, video arrow and map."""
        if self._trusted is None:
            return None
        y, o0 = self._trusted
        odo = self._odo_yaw()
        if self.sign_conflict:                     # don't add a turn we can't trust: camera only
            p = self.pose
            if p and p.get("ok") and p.get("yaw_src") in ("markers", "light", "motion", "set"):
                return float(p["yaw"])
            return math.atan2(math.sin(y), math.cos(y))
        if self.gyro_fusion and odo is not None and o0 is not None:
            y += self.yaw_sign * math.atan2(math.sin(odo - o0), math.cos(odo - o0))
        return math.atan2(math.sin(y), math.cos(y))

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
        tx, ty, face = self.to_phys(tx, ty, face)                    # steer in the physical frame
        self.dock_state = "driving to the charger"
        end, phase = time.time() + timeout, "drive"
        best_d, best_t = math.inf, time.time()            # stalled (no headway in 12 s): let Vector's own docking finish
        spin, odo_prev = 0.0, self._odo_yaw()
        self.route = {"target": list(self.from_phys(tx, ty, 0.0)[:2]), "charger": [c["x"], c["y"]], "phase": phase}
        try:
            while time.time() < end:
                await asyncio.sleep(0.1)
                if b.estop or b.frozen or b.mode != "home":
                    return False
                p, yaw = self.pose, self.heading()
                if not p or not p.get("ok") or p.get("x") is None or time.time() - p["t"] > LOST_S or yaw is None:
                    await b.body.drive(0, 0, "arena")
                    continue
                x, y, yaw = self.to_phys(float(p["x"]), float(p["y"]), yaw)
                odo = self._odo_yaw()
                if odo is not None and odo_prev is not None:
                    spin += abs(math.atan2(math.sin(odo - odo_prev), math.cos(odo - odo_prev)))
                odo_prev = odo
                if self.sign_conflict or spin > SPIN_LIMIT:
                    await b.body.drive(0, 0, "arena")
                    self.dock_state = ("stopped: camera and gyro disagree on which way it turns" if self.sign_conflict
                                       else "stopped: turned over 1.25 times without getting closer")
                    log.warning("arena route: %s", self.dock_state)
                    return False
                if phase == "drive":
                    d = math.hypot(tx - x, ty - y)
                    if d < best_d - 0.02:
                        best_d, best_t, spin = d, time.time(), 0.0
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
            return self.brain.link.why_not_connected()
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
