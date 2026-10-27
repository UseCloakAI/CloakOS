"""Sim: MuJoCo physics for Vector, in its own thread, in real time.

Everything that talks to the outside is thread-safe and cheap:
  commands in   set_wheels / set_head / set_lift / ...  (plain attribute writes)
  sensors out   snapshot  (a dict swapped atomically ~60x/s)
  camera out    frame (JPEG bytes) + frame_id, degraded to look like Vector's camera
  viewer out    scene() once per world, poses() every frame
"""
from __future__ import annotations

import io
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from . import rooms
from .vector_model import HEAD_RANGE, LIFT_RANGE, MAX_WHEEL_MMPS, WHEEL_R, lift_angle, lift_height_mm

log = logging.getLogger(__name__)

CAM_W, CAM_H = 640, 360           # what Vector's camera reports (rendering can be smaller)
SURFACES = ("floor", "table", "rug")
WHEELS = ("lf", "lm", "lb", "rf", "rm", "rb")
GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def _gl_backend() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"


_gl_backend()      # before mujoco is first imported: it reads MUJOCO_GL once


class Sim:
    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("sim", {}) or {}
        self.render_w = int(c.get("render_width", 640))
        self.render_h = int(c.get("render_height", 360))
        self.fps = float(c.get("camera_fps", 15))
        self.jpeg_q = int(c.get("jpeg_quality", 40))
        self.noise = float(c.get("camera_noise", 4.0))
        self.latency = float(c.get("camera_latency_s", 0.1))
        self.camera_on = bool(c.get("camera", True))
        self.world = c.get("world", "room")
        self.assets = None
        if c.get("assets", True):
            from .assets import Assets
            self.assets = Assets(c.get("asset_dir", "data/sim_assets"), online=bool(c.get("download", True)))
        self.seed: int | None = c.get("seed")
        self.version = 0
        self.scene_info: rooms.Scene | None = None
        self.snapshot: dict[str, Any] = {}
        self.frame: bytes | None = None
        self.frame_id = 0
        self.render_error: str | None = None
        self.button_pressed = False
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._run = False
        self._load_req: tuple[str, int | None] | None = None
        self._loaded = threading.Event()
        self._req_id = self._done_id = 0       # a load() only returns once *its* world is in
        # commands (written from asyncio, read by the physics thread)
        self.wheels = (0.0, 0.0)            # target mm/s
        self.wheel_accel = 0.0              # mm/s^2, 0 = as fast as the motors go
        self.head_target = 0.0              # rad
        self.head_rate = 0.0                # rad/s, overrides target while non-zero
        self.lift_target = lift_angle(32)
        self.lift_rate = 0.0
        self._wheel_cmd = np.zeros(2)       # ramped wheel speeds actually sent
        self._hold_until = 0.0              # picked up by an invisible hand until then
        self._hold_pos: np.ndarray | None = None
        self._push: tuple[float, np.ndarray] | None = None
        self._taps: deque[tuple[float, bool]] = deque()
        self._frames: deque[tuple[float, bytes]] = deque()
        self._air_since: float | None = None
        self._m = self._d = None
        # Vector's firmware stops forward driving the moment a front cliff sensor trips
        # (unless the SDK holds override priority); emulate that so the brain sees what it would
        self.cliff_guard = bool(c.get("cliff_guard", True))
        # an optional fast controller run inside the physics loop (sim-only skills like balancing)
        self.controller = None
        self.controller_hz = 50.0
        # mapping: depth + camera pose + what's in view, rendered only when someone wants it
        self.want_depth = False
        self.depth_hz = 2.0
        self.depth_frame: dict | None = None
        self._next_depth = 0.0
        self._edge = False
        self._n = 0

    # ── lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._run = True
        if self.scene_info is None:
            self._load_req = (self.world, self.seed, 0)
        self._thread = threading.Thread(target=self._loop, name="sim", daemon=True)
        self._thread.start()
        self._loaded.wait(10)

    def stop(self) -> None:
        self._run = False
        if self._thread:
            self._thread.join(5)

    def load(self, world: str | None = None, seed: int | None = None, wait: bool = True) -> rooms.Scene | None:
        """Swap in a new world (random seed if None). Happens on the physics thread."""
        with self._lock:
            self._req_id += 1
            me = self._req_id
            self._load_req = (world or self.world, seed, me)
        if not (self._thread and self._thread.is_alive()):
            self._do_load()
        elif wait:
            end = time.monotonic() + 90             # first use of new scanned objects downloads them
            while self._done_id < me and time.monotonic() < end:
                self._loaded.wait(0.2)
                self._loaded.clear()
        return self.scene_info

    def _do_load(self) -> None:
        import mujoco

        world, seed, rid = self._load_req or (self.world, None, 0)
        self._load_req = None
        sc = rooms.build(world, seed, self.assets)
        m = mujoco.MjModel.from_xml_string(sc.xml)
        d = mujoco.MjData(m)
        if sc.start_pitch:
            self._tip_back(m, d, sc)
        self._settle_objects(m, d, sc)
        mujoco.mj_forward(m, d)
        with self._lock:
            self._m, self._d, self.scene_info = m, d, sc
            self.world, self.seed = sc.world, sc.seed
            self._ids(m)
            self.wheels, self._wheel_cmd = (0.0, 0.0), np.zeros(2)
            self.head_target, self.head_rate = 0.0, 0.0
            self.lift_target, self.lift_rate = lift_angle(32), 0.0
            self._hold_until, self._hold_pos, self._push, self._air_since = 0.0, None, None, None
            self._frames.clear()
            self.version += 1
        self._update_snapshot()
        self._done_id = max(self._done_id, rid)
        self._loaded.set()
        log.info("sim world: %s (seed %d)", sc.world, sc.seed)

    @staticmethod
    def _settle_objects(m, d, sc: rooms.Scene) -> None:
        """Rest every loose object on the floor by its real (mesh) bounds, then let physics settle it."""
        import mujoco

        if not sc.objects:
            return
        mujoco.mj_forward(m, d)
        for name in sc.objects:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0 or m.body_jntnum[b] == 0:
                continue
            lowest = math.inf
            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
                c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                R = d.geom_xmat[g].reshape(3, 3)
                for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                    z = (d.geom_xpos[g] + R @ (c + corner * h))[2]
                    lowest = min(lowest, z)
            a = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[a + 2] += sc.floor_z + 0.001 - lowest
        robot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "vector")
        keep = d.qpos[m.jnt_qposadr[robot]:m.jnt_qposadr[robot] + 7].copy()
        for _ in range(150):                                  # 0.3 s: objects come to rest
            mujoco.mj_step(m, d)
            d.qpos[m.jnt_qposadr[robot]:m.jnt_qposadr[robot] + 7] = keep
            d.qvel[m.jnt_dofadr[robot]:m.jnt_dofadr[robot] + 6] = 0
        d.time = 0.0

    @staticmethod
    def _tip_back(m, d, sc: rooms.Scene) -> None:
        """Pitch the robot nose-up about its rear axle so it rests on the back tread wheels."""
        import mujoco

        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "vector")
        a = m.jnt_qposadr[jid]
        yaw = sc.spawn[3]
        p = sc.start_pitch
        q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        q_pitch = np.array([math.cos(-p / 2), 0, math.sin(-p / 2), 0])     # nose up = negative about y
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_yaw, q_pitch)
        ax = 0.0215
        back = np.array([-ax * math.cos(yaw), -ax * math.sin(yaw)])
        # rotate the body about the rear axle: origin moves up/back
        d.qpos[a:a + 3] = [sc.spawn[0] + back[0] + ax * math.cos(p) * math.cos(yaw),
                           sc.spawn[1] + back[1] + ax * math.cos(p) * math.sin(yaw),
                           sc.floor_z + ax * math.sin(p) + 0.001]
        d.qpos[a + 3:a + 7] = q

    def _ids(self, m) -> None:
        import mujoco

        n = lambda t, s: mujoco.mj_name2id(m, t, s)  # noqa: E731
        J, A, B, S, G = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_BODY,
                         mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_GEOM)
        self._bid = n(B, "vector")
        self._qadr = m.jnt_qposadr[n(J, "vector")]
        self._vadr = m.jnt_dofadr[n(J, "vector")]
        self._wheel_act = {k: n(A, f"wheel_{k}") for k in WHEELS}
        self._wheel_dof = {k: m.jnt_dofadr[n(J, f"wheel_{k}")] for k in WHEELS}
        self._head_act, self._lift_act = n(A, "head"), n(A, "lift")
        self._head_q, self._lift_q = m.jnt_qposadr[n(J, "head")], m.jnt_qposadr[n(J, "lift")]
        self._sites = {k: n(S, k) for k in ("prox", "cliff_fl", "cliff_fr", "cliff_bl", "cliff_br", "imu")}
        self._face_site = n(S, "face")
        self._person_head = n(G, "person_head")
        self._surface = {n(G, s) for s in SURFACES} - {-1}
        robot = set()
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            while b > 0:
                if b == self._bid:
                    robot.add(g)
                    break
                b = m.body_parentid[b]
        self._robot_geoms = robot
        self._wheel_geoms = {n(G, f"wheel_{k}") for k in WHEELS}
        self._accel = m.sensor_adr[n(mujoco.mjtObj.mjOBJ_SENSOR, "accel")]
        self._gyro = m.sensor_adr[n(mujoco.mjtObj.mjOBJ_SENSOR, "gyro")]
        self._cam = n(mujoco.mjtObj.mjOBJ_CAMERA, "eye")
        self._rays = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)   # every group but the robot's own (3)

    # ── commands (any thread) ──────────────────────────────────

    def set_wheels(self, left: float, right: float, accel: float = 0.0) -> None:
        c = lambda v: max(-MAX_WHEEL_MMPS, min(MAX_WHEEL_MMPS, float(v)))  # noqa: E731
        self.wheels, self.wheel_accel = (c(left), c(right)), float(accel)

    def set_head(self, angle: float | None = None, rate: float = 0.0) -> None:
        if angle is not None:
            self.head_target = max(math.radians(HEAD_RANGE[0]), min(math.radians(HEAD_RANGE[1]), angle))
        self.head_rate = rate

    def set_lift(self, angle: float | None = None, rate: float = 0.0) -> None:
        if angle is not None:
            self.lift_target = max(LIFT_RANGE[0], min(LIFT_RANGE[1], angle))
        self.lift_rate = rate

    def stop_motors(self) -> None:
        self.wheels, self.head_rate, self.lift_rate = (0.0, 0.0), 0.0, 0.0
        s = self.snapshot
        if s:
            self.head_target, self.lift_target = s["head_angle"], lift_angle(s["lift_mm"])

    def pick_up(self, seconds: float = 2.5, height: float = 0.12) -> None:
        """An invisible hand lifts Vector straight up, holds it, then sets it down gently."""
        with self._lock:
            if self._d is None:
                return
            self._hold_pos = self._d.qpos[self._qadr:self._qadr + 3].copy()
            self._hold_pos[2] += height
            self._hold_until = time.monotonic() + seconds

    def knock(self, strength: float = 1.0, direction: float | None = None) -> None:
        """A shove at head height. direction: radians in the world frame (random if None)."""
        a = direction if direction is not None else np.random.uniform(-math.pi, math.pi)
        f = 2.2 * strength
        t = self._d.time if self._d is not None else 0.0      # sim time: correct at any speed
        self._push = (t + 0.05, np.array([f * math.cos(a), f * math.sin(a), 0.0, 0.0, 0.0, 0.0]))

    def tap_button(self, n: int = 1) -> None:
        t = time.monotonic()
        for i in range(n):
            self._taps.append((t + i * 0.3, True))
            self._taps.append((t + i * 0.3 + 0.12, False))

    def move_person(self, x: float, y: float) -> None:
        import mujoco

        with self._lock:
            if self._m is None:
                return
            b = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, "person")
            if b >= 0:
                self._d.mocap_pos[self._m.body_mocapid[b]][:2] = [x, y]
                self.scene_info.person = (x, y)

    # ── physics thread ─────────────────────────────────────────

    def _loop(self) -> None:
        import mujoco

        renderer, rendered_for = None, None
        next_frame = next_snap = next_ctrl = 0.0
        last = time.monotonic()
        while self._run:
            if self._load_req is not None or self._m is None:
                try:
                    self._do_load()
                except Exception:
                    log.exception("sim world failed to load")
                    self._loaded.set()
                    time.sleep(1)
                    continue
                last = time.monotonic()
            now = time.monotonic()
            with self._lock:
                m, d = self._m, self._d
                steps = min(25, int((now - last) / m.opt.timestep))
                for _ in range(steps):
                    self._control(m, d, now)
                    mujoco.mj_step(m, d)
                last += steps * m.opt.timestep
                if now - last > 0.25:          # fell far behind (debugger, sleep): don't try to catch up
                    last = now
            ctrl = self.controller
            if ctrl is not None and now >= next_ctrl:
                self._update_snapshot()
                try:
                    self.set_wheels(*ctrl(self.snapshot))
                except Exception:
                    log.exception("sim controller failed; removing it")
                    self.controller = None
                next_ctrl = now + 1 / self.controller_hz
            if now >= next_snap:
                self._update_snapshot()
                next_snap = now + 1 / 60
            if self.camera_on and now >= next_frame:
                next_frame = now + 1 / self.fps
                if rendered_for != self.version:
                    renderer, rendered_for = self._make_renderer(renderer), self.version
                if renderer is not None:
                    self._capture(renderer, now)
                    if self.want_depth and now >= self._next_depth:
                        self._next_depth = now + 1 / self.depth_hz
                        self._capture_depth(renderer)
            self._publish_frame(now)
            time.sleep(0.002)
        if renderer is not None:
            renderer.close()

    def _make_renderer(self, old):
        import mujoco

        if old is not None:
            old.close()
        try:
            r = mujoco.Renderer(self._m, self.render_h, self.render_w)
            self.render_error = None
            return r
        except Exception as e:
            self.render_error = f"camera rendering unavailable: {e}"
            log.warning(self.render_error)
            self.camera_on = False
            return None

    def _control(self, m, d, now: float) -> None:
        dt = m.opt.timestep
        target = np.array(self.wheels)
        if self.wheel_accel > 0:
            step = self.wheel_accel * dt
            self._wheel_cmd += np.clip(target - self._wheel_cmd, -step, step)
        else:
            self._wheel_cmd = target
        self._n += 1
        if self.cliff_guard and self._n % 5 == 0:     # every 10 ms, like the firmware's sensor loop
            R = d.xmat[self._bid].reshape(3, 3)
            down = -R[:, 2]
            up = R[2, 2] > 0.7
            self._edge = up and any(self._ray(m, d, self._sites[k], down) > 0.02 for k in ("cliff_fl", "cliff_fr"))
            self._edge_back = up and any(self._ray(m, d, self._sites[k], down) > 0.02 for k in ("cliff_bl", "cliff_br"))
        if self._hold_pos is None:
            if self._edge:
                self._wheel_cmd = np.minimum(self._wheel_cmd, 0.0)    # forward blocked, reversing allowed
            if getattr(self, "_edge_back", False):
                self._wheel_cmd = np.maximum(self._wheel_cmd, 0.0)    # the rear sensors stop it backing off
        l, r = self._wheel_cmd / 1000.0 / WHEEL_R
        for k in WHEELS:
            d.ctrl[self._wheel_act[k]] = l if k[0] == "l" else r
        if self.head_rate:
            self.set_head(self.head_target + self.head_rate * dt, self.head_rate)
        if self.lift_rate:
            self.set_lift(self.lift_target + self.lift_rate * dt, self.lift_rate)
        d.ctrl[self._head_act] = self.head_target
        d.ctrl[self._lift_act] = self.lift_target
        # invisible hand
        if self._hold_pos is not None:
            a, v = self._qadr, self._vadr
            if now < self._hold_until:
                d.qpos[a:a + 3] += np.clip(self._hold_pos - d.qpos[a:a + 3], -0.0006, 0.0006)
                d.qvel[v:v + 6] = 0
            else:
                self._hold_pos[2] -= 0.0006                     # set it back down slowly
                floor = (self.scene_info.floor_z if self.scene_info else 0.0) + 0.0005
                if self._hold_pos[2] <= floor or d.qpos[a + 2] <= floor:
                    self._hold_pos = None
                else:
                    d.qpos[a:a + 3] = self._hold_pos
                    d.qvel[v:v + 6] = 0
        if self._push is not None:
            until, f = self._push
            d.xfrc_applied[self._bid] = f if d.time < until else 0
            if d.time >= until:
                self._push = None
        while self._taps and self._taps[0][0] <= now:
            self.button_pressed = self._taps.popleft()[1]

    # ── direct stepping (RL: no thread, as fast as the CPU goes) ──

    def step(self, seconds: float) -> None:
        """Advance physics synchronously (only when the sim thread isn't running)."""
        import mujoco

        m, d = self._m, self._d
        now = time.monotonic()
        for _ in range(max(1, int(round(seconds / m.opt.timestep)))):
            self._control(m, d, now)
            mujoco.mj_step(m, d)
        self._update_snapshot()

    def place_robot(self, x: float, y: float, yaw: float, pitch: float = 0.0, tumble: tuple[float, float] | None = None) -> None:
        """Teleport Vector (standing, tipped back by `pitch`, or dropped at a `tumble` (pitch, roll)
        orientation from a few cm up so it lands however it lands) and zero its motion."""
        import mujoco

        m, d, sc = self._m, self._d, self.scene_info
        a, v = self._qadr, self._vadr
        d.qvel[:] = 0
        sc.spawn = (x, y, sc.floor_z, yaw)
        sc.start_pitch = pitch
        if tumble:
            tp, tr = tumble
            q = np.zeros(4)
            qy, qp, qr = (np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
                          np.array([math.cos(-tp / 2), 0, math.sin(-tp / 2), 0]),
                          np.array([math.cos(tr / 2), math.sin(tr / 2), 0, 0]))
            mujoco.mju_mulQuat(q, qy, qp)
            mujoco.mju_mulQuat(q, q.copy(), qr)
            d.qpos[a:a + 3] = [x, y, sc.floor_z + 0.06]
            d.qpos[a + 3:a + 7] = q
        elif pitch:
            self._tip_back(m, d, sc)
        else:
            d.qpos[a:a + 3] = [x, y, sc.floor_z + 0.0005]
            d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.wheels, self._wheel_cmd = (0.0, 0.0), np.zeros(2)
        self.head_target, self.head_rate, self.lift_rate = 0.0, 0.0, 0.0
        d.qpos[self._head_q] = 0.0
        self._hold_pos, self._push, self._air_since, self._edge, self._n = None, None, None, False, 0
        d.ctrl[:] = 0
        d.act[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        self._update_snapshot()

    def rays(self, n: int = 9, fov: float = math.radians(120), max_range: float = 1.0) -> np.ndarray:
        """Distances (m) along a horizontal fan from the robot's nose: a stand-in for depth."""
        m, d = self._m, self._d
        R = d.xmat[self._bid].reshape(3, 3)
        fwd, left = R[:, 0].copy(), R[:, 1].copy()
        fwd[2] = left[2] = 0
        fwd /= max(1e-6, np.linalg.norm(fwd))
        left /= max(1e-6, np.linalg.norm(left))
        out = np.empty(n, np.float32)
        for i, a in enumerate(np.linspace(-fov / 2, fov / 2, n)):
            out[i] = min(max_range, self._ray(m, d, self._sites["prox"], math.cos(a) * fwd + math.sin(a) * left))
        return out

    # ── sensors ────────────────────────────────────────────────

    def _ray(self, m, d, site: int, direction: np.ndarray) -> float:
        import mujoco

        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(m, d, d.site_xpos[site], direction, self._rays, 1, -1, gid)
        return dist if dist >= 0 else math.inf

    def _update_snapshot(self) -> None:
        with self._lock:
            m, d = self._m, self._d
            if m is None:
                return
            a, v = self._qadr, self._vadr
            pos = d.qpos[a:a + 3].copy()
            quat = d.qpos[a + 3:a + 7].copy()
            R = d.xmat[self._bid].reshape(3, 3)
            yaw = math.atan2(R[1, 0], R[0, 0])
            pitch = math.asin(max(-1.0, min(1.0, R[2, 0])))         # + = nose up
            roll = math.atan2(R[2, 1], R[2, 2])
            up = float(R[2, 2])
            fwd = R[:, 0]
            prox = self._ray(m, d, self._sites["prox"], fwd)
            down = -R[:, 2]
            cliffs = {k: self._ray(m, d, self._sites[k], down) for k in ("cliff_fl", "cliff_fr", "cliff_bl", "cliff_br")}
            cliff = any(cliffs[k] > 0.02 for k in ("cliff_fl", "cliff_fr"))
            wheel_ground = bump = body_ground = False
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = c.geom1, c.geom2
                mine, other = (g1, g2) if g1 in self._robot_geoms else (g2, g1) if g2 in self._robot_geoms else (None, None)
                if mine is None:
                    continue
                if other in self._surface:
                    wheel_ground |= mine in self._wheel_geoms
                    body_ground |= mine not in self._wheel_geoms
                else:
                    bump = True
            now = time.monotonic()
            airborne = all(c > 0.02 for c in cliffs.values()) and not wheel_ground
            if airborne:
                self._air_since = self._air_since or now
            else:
                self._air_since = None
            held = self._hold_pos is not None and now < self._hold_until
            wl = d.qvel[self._wheel_dof["lm"]] * WHEEL_R * 1000
            wr = d.qvel[self._wheel_dof["rm"]] * WHEEL_R * 1000
            vel = d.qvel[v:v + 3]
            person = None
            if self._face_site >= 0:
                person = self._see_face(m, d)
            self.snapshot = {
                "t": float(d.time), "pos": pos.tolist(), "quat": quat.tolist(), "yaw": yaw, "pitch": pitch, "roll": roll,
                "upright": up, "speed": float(math.hypot(vel[0], vel[1])), "vz": float(vel[2]),
                "left_mmps": float(wl), "right_mmps": float(wr),
                "head_angle": float(d.qpos[self._head_q]), "lift_mm": lift_height_mm(float(d.qpos[self._lift_q])),
                "prox_m": prox if prox < 0.4 else None, "cliff": cliff and not airborne,
                "picked_up": held or (self._air_since is not None and now - self._air_since > 0.25),
                "falling": airborne and vel[2] < -0.3 and not held,
                "bump": bump, "wheel_ground": wheel_ground, "body_ground": body_ground,
                "accel": (d.sensordata[self._accel:self._accel + 3] * 1000).tolist(),
                "gyro": d.sensordata[self._gyro:self._gyro + 3].tolist(),
                "button": self.button_pressed, "face": person,
                "moving": bool(abs(wl) + abs(wr) > 5),
            }

    def _see_face(self, m, d) -> dict | None:
        """Where the person's face lands in Vector's camera, if it can see it (in view, unblocked, facing it)."""
        import mujoco

        p = d.site_xpos[self._face_site]
        c = d.cam_xpos[self._cam]
        R = d.cam_xmat[self._cam].reshape(3, 3)
        rel = R.T @ (p - c)
        depth = -rel[2]
        if not 0.08 < depth < 1.6:
            return None
        f = (CAM_H / 2) / math.tan(math.radians(m.cam_fovy[self._cam]) / 2)
        x = CAM_W / 2 + f * rel[0] / depth
        y = CAM_H / 2 - f * rel[1] / depth
        if not (0 <= x <= CAM_W and 0 <= y <= CAM_H):
            return None
        body = m.site_bodyid[self._face_site]
        facing = d.xmat[body].reshape(3, 3)[:, 0]
        to_cam = (c - p) / max(1e-6, np.linalg.norm(c - p))
        if facing @ to_cam < 0.15:
            return None
        gid = np.zeros(1, dtype=np.int32)
        ray = (p - c) / np.linalg.norm(p - c)
        hit = mujoco.mj_ray(m, d, c, ray, self._rays, 1, self._bid, gid)
        if hit >= 0 and gid[0] != self._person_head and hit < np.linalg.norm(p - c) - 0.11:
            return None
        w = f * 0.15 / depth
        return {"x": x - w / 2, "y": y - w * 0.6, "w": w, "h": w * 1.2, "depth": depth}

    # ── camera ─────────────────────────────────────────────────

    def _capture(self, renderer, now: float) -> None:
        from PIL import Image, ImageFilter

        with self._lock:
            renderer.update_scene(self._d, camera="eye")
            px = renderer.render()
        img = px.astype(np.float32)
        if self.noise:
            img += np.random.normal(0, self.noise, img.shape)
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if im.size != (CAM_W, CAM_H):
            im = im.resize((CAM_W, CAM_H), Image.BILINEAR)
        im = im.filter(ImageFilter.GaussianBlur(0.7))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=self.jpeg_q)
        self._frames.append((now, buf.getvalue()))

    def _capture_depth(self, renderer) -> None:
        """Exact depth + camera pose + visible labelled objects, for the 3D map."""
        with self._lock:
            m, d, sc = self._m, self._d, self.scene_info
            renderer.update_scene(d, camera="eye")
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(d, camera="eye")
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            M = d.cam_xmat[self._cam].reshape(3, 3)
            R = np.stack([-M[:, 2], -M[:, 0], M[:, 1]], 1)          # forward, left, up
            pos = d.cam_xpos[self._cam].copy()
            dets = self._visible_objects(m, d, sc, pos, R, depth.shape[1], depth.shape[0])
        self.depth_frame = {"t": time.time(), "depth": depth.astype(np.float32), "rgb": rgb, "pos": pos, "R": R,
                            "detections": dets, "floor_z": sc.floor_z, "key": f"sim-{sc.world}-{sc.seed}"}

    def _visible_objects(self, m, d, sc, cam, R, w, h) -> list[dict]:
        import mujoco

        f = (h / 2) / math.tan(math.radians(m.cam_fovy[self._cam]) / 2)
        things = [(name, sc.labels.get(name, name)) for name in sc.objects]
        if sc.person:
            things.append(("person", "person"))
        out = []
        gid = np.zeros(1, dtype=np.int32)
        for body_name, label in things:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if b < 0:
                continue
            p = d.xpos[b] + (np.array([0, 0, 0.4]) if body_name == "person" else 0)
            rel = R.T @ (p - cam)
            if not 0.05 < rel[0] < 2.5:
                continue
            u, v = w / 2 - f * rel[1] / rel[0], h / 2 - f * rel[2] / rel[0]
            if not (0 <= u < w and 0 <= v < h):
                continue
            dist = float(np.linalg.norm(p - cam))
            hit = mujoco.mj_ray(m, d, cam, (p - cam) / dist, self._rays, 1, self._bid, gid)
            if hit >= 0 and m.geom_bodyid[gid[0]] != b and hit < dist - 0.04:
                continue
            r = f * (0.25 if body_name == "person" else 0.06) / rel[0]
            out.append({"label": label, "score": 1.0, "box": (u - r, v - r, u + r, v + r), "id": body_name})
        return out

    def _publish_frame(self, now: float) -> None:
        latest = None
        while self._frames and self._frames[0][0] <= now - self.latency:
            latest = self._frames.popleft()[1]
        if latest is not None:
            self.frame = latest
            self.frame_id += 1

    # ── viewer ─────────────────────────────────────────────────

    def scene(self) -> dict[str, Any]:
        """Static description for the 3D viewer: bodies, their geoms, materials (meshes and
        textures are fetched separately via mesh_data() / texture_png())."""
        import mujoco

        with self._lock:
            m, sc = self._m, self.scene_info
            if m is None:
                return {"version": 0, "geoms": []}
            geoms = []
            for g in range(m.ngeom):
                t = GEOM_TYPES.get(int(m.geom_type[g]))
                if t is None or m.geom_rgba[g][3] == 0:
                    continue
                mat = int(m.geom_matid[g])
                info = {
                    "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
                    "body": int(m.geom_bodyid[g]), "type": t,
                    "size": [round(float(x), 5) for x in m.geom_size[g]],
                    "pos": [round(float(x), 5) for x in m.geom_pos[g]],
                    "quat": [round(float(x), 5) for x in m.geom_quat[g]],
                    "rgba": [round(float(x), 3) for x in (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g])],
                }
                if t == "mesh":
                    info["mesh"] = int(m.geom_dataid[g])
                if mat >= 0:
                    tex = int(m.mat_texid[mat][1]) if m.mat_texid.ndim == 2 else int(m.mat_texid[mat])
                    if tex < 0 and m.mat_texid.ndim == 2:
                        tex = int(max(m.mat_texid[mat]))
                    info["mat"] = {"tex": tex, "repeat": [float(x) for x in m.mat_texrepeat[mat]],
                                   "uniform": bool(m.mat_texuniform[mat]), "emission": float(m.mat_emission[mat]),
                                   "specular": float(m.mat_specular[mat]), "shininess": float(m.mat_shininess[mat]),
                                   "reflectance": float(m.mat_reflectance[mat])}
                geoms.append(info)
            return {"version": self.version, "world": sc.world, "seed": sc.seed, "room": sc.room,
                    "nbody": int(m.nbody), "robot": int(self._bid), "geoms": geoms,
                    "objects": sc.objects, "labels": sc.labels, "person": sc.person, "goal": sc.goal}

    def mesh_data(self, mesh: int) -> bytes | None:
        """One mesh as flat triangles for three.js: u32 count, then positions, normals, uvs (f32)."""
        with self._lock:
            m = self._m
            if m is None or not 0 <= mesh < m.nmesh:
                return None
            va, fa, fn = m.mesh_vertadr[mesh], m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            faces = m.mesh_face[fa:fa + fn]
            pos = m.mesh_vert[va:va + m.mesh_vertnum[mesh]][faces].reshape(-1, 3)
            na = m.mesh_normaladr[mesh]
            nrm = m.mesh_normal[na:na + m.mesh_normalnum[mesh]][m.mesh_facenormal[fa:fa + fn]].reshape(-1, 3) \
                if m.mesh_normalnum[mesh] else np.zeros_like(pos)
            ta, tn = m.mesh_texcoordadr[mesh], m.mesh_texcoordnum[mesh]
            uv = m.mesh_texcoord[ta:ta + tn][m.mesh_facetexcoord[fa:fa + fn]].reshape(-1, 2) if ta >= 0 and tn else np.zeros((0, 2))
        head = np.array([len(pos), len(uv)], dtype=np.uint32).tobytes()
        return head + pos.astype(np.float32).tobytes() + nrm.astype(np.float32).tobytes() + uv.astype(np.float32).tobytes()

    def texture_png(self, tex: int) -> bytes | None:
        from PIL import Image

        with self._lock:
            m = self._m
            if m is None or not 0 <= tex < m.ntex:
                return None
            w, h, c = int(m.tex_width[tex]), int(m.tex_height[tex]), int(m.tex_nchannel[tex])
            a = m.tex_adr[tex]
            px = np.array(m.tex_data[a:a + w * h * c]).reshape(h, w, c)
        buf = io.BytesIO()
        Image.fromarray(px[..., :3] if c >= 3 else px[..., 0]).save(buf, "PNG")
        return buf.getvalue()

    def poses(self) -> list[float]:
        """xpos + xquat for every body, flat: [x, y, z, qw, qx, qy, qz] * nbody."""
        with self._lock:
            if self._d is None:
                return []
            return np.round(np.hstack([self._d.xpos, self._d.xquat]), 4).ravel().tolist()
