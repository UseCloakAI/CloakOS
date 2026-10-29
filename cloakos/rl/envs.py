"""Gymnasium environments on the MuJoCo Vector: what reinforcement learning trains in.

Tasks
  goto     drive to a point in a cluttered room, around obstacles, without bumping or falling
  wander   roam the room, covering new ground, without bumping into things
  recover  knocked onto its back or nose: get back on its treads with the wheels and lift

(A two-wheel "wheelie" balance isn't in the list: Vector's body reaches past its rear axle,
below axle height, so tipping back lands it on its rear before the treads can balance it.
That's true of the real robot too.)

Sensors (obs) come in two flavours:
  robot    only what the real Vector reports: odometry-relative goal, wheel speeds, the single
           forward distance sensor, cliff, tilt.
  map      adds 11 range readings around the robot (180°): how far to the nearest obstacle in
           each direction. In training they come from the sim's geometry, with noise and gaps;
           on the real robot they're read from the live 3D map. Both run on the real robot.

Actions are wheel speeds in -1..1 (x max speed), the same as Body.drive().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:          # the brain itself doesn't need gymnasium; only training does
    gym = None

from ..sim.engine import Sim

TASKS = {
    "goto": "Drive to a point, avoid obstacles and edges",
    "wander": "Roam the room without bumping into things",
    "desk": "Desk: roam a random desk, never fall, avoid things (teacher, sees exact edges in the sim)",
    "desk-pixels": "Desk from the camera: learns from the desk teacher, runs on the real robot",
    "recover": "Get back on its treads after a fall (experimental: train long)",
}
SENSORS = {"map": "Map ranges + robot sensors (recommended)", "robot": "Robot sensors only"}
ALIASES = {"explore": "wander", "rays": "map"}       # names used by older policies
RAY_FOV = math.radians(180)
RAY_MAX = 1.0
DESK_REWARD_V = 2               # bump when the desk reward changes: older desk drivers get retrained (v2: steer clear of things)
REAL_LAG_STEPS = (3, 5)          # command -> motion, in 0.1 s steps (calibration: 0.41 s)
REAL_WHEEL_GAIN = (0.82, 0.92)   # wheels deliver this share of the asked speed (calibration: 0.86)


@dataclass
class Spec:
    task: str
    sensors: str = "robot"
    control_hz: float = 10.0
    max_speed: float = 150.0          # mm/s at action 1.0
    episode_s: float = 30.0
    n_rays: int = 11

    @property
    def obs_size(self) -> int:
        base = {"goto": 9, "wander": 7, "desk": 7, "recover": 12}[self.task]
        return base + (self.n_rays if self.sensors == "map" else 0)


def make_spec(task: str, sensors: str = "map") -> Spec:
    task, sensors = ALIASES.get(task, task), ALIASES.get(sensors, sensors)
    if task == "desk":                 # the teacher: always sees exact edges + things (sim-only)
        return Spec("desk", "map", episode_s=40.0)
    if task == "recover":
        return Spec("recover", "robot", control_hz=10.0, max_speed=200.0, episode_s=10.0)
    return Spec(task, sensors if sensors in SENSORS else "map", episode_s=40.0 if task == "wander" else 30.0)


class EdgeReflex:
    """CloakOS's edge reflex exactly as the brain runs it on the robot (connection/robot.py
    _edge_reflex), so training sees what really happens at an edge. The robot runs at override
    priority, which turns Vector's own edge stop off: this reflex is the only safety net.

    On a new edge: after the Wi-Fi delay, move away from where it was heading (reverse if it was
    going forward) at 75 mm/s for 0.7-1.3 s (faster in = further out), up to 3 tries until the
    sensor clears, then stop. While it runs it owns the wheels; the policy's commands are ignored."""
    SPEED = 75.0

    def __init__(self, delay_steps: int, hz: float) -> None:
        self.delay, self.hz = max(1, delay_steps), hz
        self.t, self.was, self.pending, self.until, self.tries, self.v, self.dur = 0, False, None, 0, 0, 0.0, 0

    def step(self, s: dict) -> float | None:
        """Call once per control step with the sensor snapshot; returns the wheel speed (mm/s, both
        sides) while the reflex owns the wheels, else None."""
        self.t += 1
        edge = bool(s.get("cliff"))
        if edge and not self.was and self.pending is None and not self.tries:
            self.pending = (self.t + self.delay, (s["left_mmps"] + s["right_mmps"]) / 2)
        self.was = edge
        if self.pending and self.t >= self.pending[0]:
            fwd = self.pending[1]
            self.pending = None
            self.v = (1.0 if fwd < -10 else -1.0) * self.SPEED
            self.dur = max(1, round((0.7 + min(0.6, abs(fwd) / 150 * 0.6)) * self.hz))
            self.tries, self.until = 1, self.t + self.dur
        if self.tries:
            if self.t < self.until:
                return self.v
            if edge and self.tries < 3:
                self.tries += 1
                self.until = self.t + self.dur
                return self.v
            self.tries = 0
            return 0.0
        return None


def edge_rays(sn: dict, room, n: int) -> np.ndarray:
    """Range readings to a table's drop-off, treated like a wall (what the real map's edge of floor gives)."""
    hx, hy = room
    x0, y0 = sn["pos"][0] + 0.04 * math.cos(sn["yaw"]), sn["pos"][1] + 0.04 * math.sin(sn["yaw"])
    out = np.full(n, RAY_MAX, np.float32)
    for i, a in enumerate(np.linspace(-RAY_FOV / 2, RAY_FOV / 2, n)):
        dx, dy = math.cos(sn["yaw"] + a), math.sin(sn["yaw"] + a)
        t = RAY_MAX
        if dx > 1e-6: t = min(t, (hx - x0) / dx)
        elif dx < -1e-6: t = min(t, (-hx - x0) / dx)
        if dy > 1e-6: t = min(t, (hy - y0) / dy)
        elif dy < -1e-6: t = min(t, (-hy - y0) / dy)
        out[i] = max(0.0, t - 0.03)
    return out


def observe(spec: Spec, s: dict, goal: tuple[float, float] | None, rays: np.ndarray | None,
            last_action: np.ndarray) -> np.ndarray:
    """Observation vector from a sensor snapshot. Shared by training (sim) and the robot
    (cloakos.rl.policy builds the same dict from World), so the policy sees the same thing."""
    v = spec.max_speed
    prox = s.get("prox_m")
    common = [s["left_mmps"] / v, s["right_mmps"] / v, (prox if prox is not None else 0.6) / 0.6,
              1.0 if s.get("cliff") else 0.0]
    if spec.task == "goto":
        dx, dy = goal[0] - s["pos"][0], goal[1] - s["pos"][1]
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) - s["yaw"]
        o = [min(dist, 3.0) / 3.0, math.sin(ang), math.cos(ang), *common, *last_action]
    elif spec.task in ("wander", "desk"):
        o = [*common, s["upright"], *last_action]
    else:   # recover: which way it's lying, how it's moving, where the lift is
        roll = s.get("roll", 0.0)
        o = [s["upright"], math.sin(s["pitch"]), math.cos(s["pitch"]), math.sin(roll), math.cos(roll),
             *(np.asarray(s["gyro"]) / 5.0), (s["lift_mm"] - 32) / 60, s["left_mmps"] / v, s["right_mmps"] / v,
             1.0 if s.get("body_ground") else 0.0]
    if spec.sensors == "map":
        o += list(np.clip(rays, 0, RAY_MAX) / RAY_MAX) if rays is not None else [1.0] * spec.n_rays
    return np.asarray(o, dtype=np.float32)


if gym is not None:

    class VectorEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, task: str = "goto", sensors: str = "map", seed: int | None = None,
                     worlds_per_reload: int = 8, hard: list[int] | None = None) -> None:
            super().__init__()
            self.spec_ = make_spec(task, sensors)
            n = self.spec_.obs_size
            self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (3 if task == "recover" else 2,), np.float32)
            # cliff_guard off: at override priority (what the brain uses) Vector's firmware edge stop is off
            self.sim = Sim({"sim": {"camera": False, "assets": False, "cliff_guard": False}})
            self.rng = np.random.default_rng(seed)
            self.reload_every = worlds_per_reload
            self.episodes = 0
            self.goal: tuple[float, float] | None = None
            self.last_action = np.zeros(2, np.float32)
            self.t = 0
            self.visited: set[tuple[int, int]] = set()
            self.hard: list[int] = [int(x) for x in hard or []]   # world seeds earlier exams failed on

        # ── helpers ────────────────────────────────────────────

        def _free_point(self, clear: float = 0.15) -> tuple[float, float]:
            sc = self.sim.scene_info
            hx, hy = sc.room
            for _ in range(200):
                x, y = self.rng.uniform(-hx + clear, hx - clear), self.rng.uniform(-hy + clear, hy - clear)
                if all(math.hypot(x - ox, y - oy) > r + clear for ox, oy, r in sc.obstacles):
                    return x, y
            return 0.0, 0.0

        def _rays(self):
            """Sim ranges made to look like a real map: a little noise, and some gaps (never seen = far)."""
            if self.spec_.sensors != "map":
                return None
            r = self.sim.rays(self.spec_.n_rays, RAY_FOV, RAY_MAX)
            sc = self.sim.scene_info
            if sc.world in ("table", "desk"):          # a drop-off reads as a wall, like the real map's edge of floor
                r = np.minimum(r, self._edge_rays(sc.room))
            if self.spec_.task == "desk":              # the teacher's privileged view: exact, a hair of noise
                return np.clip(r * self.rng.normal(1.0, 0.01, r.shape), 0.0, RAY_MAX).astype(np.float32)
            r = r * self.rng.normal(1.0, 0.05, r.shape)
            r[self.rng.random(r.shape) < 0.12] = RAY_MAX
            return np.clip(r, 0.0, RAY_MAX).astype(np.float32)

        def _place_on_desk(self) -> None:
            """Anywhere on the desk, any heading, clear of the edge and the things on it; the head
            and camera a little different each time (no two Vectors sit or see exactly alike)."""
            sc = self.sim.scene_info
            hx, hy = sc.room
            x = y = 0.0
            for _ in range(100):
                x, y = self.rng.uniform(-hx + 0.08, hx - 0.08), self.rng.uniform(-hy + 0.08, hy - 0.08)
                if all(math.hypot(x - ox, y - oy) > r + 0.08 for ox, oy, r in sc.obstacles):
                    break
            self.sim.place_robot(float(x), float(y), float(self.rng.uniform(-math.pi, math.pi)))
            self.sim.head_target = math.radians(float(self.rng.uniform(-15, 10)))
            m = self.sim._m
            cam = m.camera("eye").id
            if not hasattr(self, "_cam0"):
                self._cam0 = (float(m.cam_fovy[cam]), m.cam_pos[cam].copy())
            m.cam_fovy[cam] = self._cam0[0] + self.rng.uniform(-4, 4)
            m.cam_pos[cam] = self._cam0[1] + self.rng.normal(0, 0.002, 3)

        def hard_world(self, wseed: int) -> tuple[str, int]:
            """An exam world (seed) for this task, rebuilt exactly as the exam built it."""
            r = np.random.default_rng(wseed)
            task = self.spec_.task
            world = "empty" if task == "recover" else "desk" if task == "desk" else str(r.choice(["room", "room", "empty", "table"]))
            return world, int(r.integers(1, 1_000_000))

        def _edge_rays(self, room) -> np.ndarray:
            return edge_rays(self.sim.snapshot, room, self.spec_.n_rays)

        def _obs(self):
            o = observe(self.spec_, self.sim.snapshot, self.goal, self._rays(), self.last_action)
            # Wi-Fi + camera lag: sometimes the brain only has the previous reading
            self._obs_q.append(o)
            return (self._obs_q[0] if len(self._obs_q) > self.obs_delay else o).copy()

        # ── gym API ────────────────────────────────────────────

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            if seed is not None:                       # same seed -> same world and start, exactly
                self.rng = np.random.default_rng(seed)
                self.episodes = 0
            task = self.spec_.task
            if self.sim.scene_info is None or self.episodes % self.reload_every == 0:
                world = "empty" if task == "recover" else "desk" if task == "desk" else str(self.rng.choice(["room", "room", "empty", "table"]))
                wseed = int(self.rng.integers(1, 1_000_000))
                if self.hard and seed is None and self.rng.random() < 0.25:
                    world, wseed = self.hard_world(int(self.rng.choice(self.hard)))
                self.sim.load(world, wseed)
            self.episodes += 1
            sc = self.sim.scene_info
            if task == "recover":                  # dropped on its back, side or nose; let it land
                # on its back or on its nose (on its side the real Vector can't right itself either)
                tumble = [(math.radians(self.rng.uniform(70, 110)), math.radians(self.rng.uniform(-25, 25))),
                          (math.radians(self.rng.uniform(-110, -70)), math.radians(self.rng.uniform(-25, 25)))][int(self.rng.integers(2))]
                for _ in range(6):                    # re-drop lucky landings: every episode is a real recovery
                    self.sim.place_robot(0.0, 0.0, float(self.rng.uniform(-math.pi, math.pi)), tumble=tumble)
                    self.sim.step(0.6)
                    if self.sim.snapshot["upright"] < 0.5:
                        break
            elif sc.world == "desk":
                self._place_on_desk()
            elif sc.world == "table":
                self.sim.place_robot(float(self.rng.uniform(-0.1, 0.1)), float(self.rng.uniform(-0.05, 0.05)),
                                     float(self.rng.uniform(-math.pi, math.pi)))
            else:
                x, y = self._free_point(0.2)
                self.sim.place_robot(x, y, float(self.rng.uniform(-math.pi, math.pi)))
            if task == "goto":
                if sc.world == "table":
                    hx, hy = sc.room
                    self.goal = (float(self.rng.uniform(-hx + 0.08, hx - 0.08)), float(self.rng.uniform(-hy + 0.08, hy - 0.08)))
                else:
                    p0 = self.sim.snapshot["pos"]
                    for _ in range(20):                      # not trivially close
                        self.goal = self._free_point(0.12)
                        if math.dist(self.goal, p0[:2]) > 0.25:
                            break
            self.last_action = np.zeros(2, np.float32)       # what the brain last asked for (it knows that, not what landed)
            self.t = 0
            self.bumps = 0
            self.visited = set()
            # latency like the real robot over Wi-Fi: commands land 0-200 ms late, readings 0-100 ms stale
            from collections import deque
            # measured on the real Vector (calibration): ~0.41 s from command to motion, wheels 86% of asked
            self.act_delay = int(self.rng.integers(REAL_LAG_STEPS[0], REAL_LAG_STEPS[1] + 1))
            self.obs_delay = int(self.rng.integers(0, 2))
            self.wheel_gain = float(self.rng.uniform(*REAL_WHEEL_GAIN))
            self._act_q = deque([np.zeros(3, np.float32)] * self.act_delay)
            self.reflex = EdgeReflex(self.act_delay - 1, self.spec_.control_hz) if task != "recover" else None
            self._obs_q = deque(maxlen=self.obs_delay + 1)
            s = self.sim.snapshot
            self._prev_dist = math.dist(s["pos"][:2], self.goal) if self.goal else 0.0
            self._prev_up, self._upright_for = s["upright"], 0
            return self._obs(), {}

        def step(self, action):
            spec, sim = self.spec_, self.sim
            a = np.clip(np.asarray(action, np.float32), -1, 1)
            jerk = float(np.sum((a[:2] - self.last_action) ** 2))
            self.last_action = a[:2].astype(np.float32)
            self._act_q.append(np.pad(a, (0, 3 - len(a))))
            a = self._act_q.popleft()[:len(a)]          # what actually reaches the robot this step
            left, right = float(a[0]) * spec.max_speed, float(a[1]) * spec.max_speed
            if spec.task == "recover":
                sim.set_lift(None, float(a[2]) * 3.0)
                sim.set_wheels(left, right, 0.0)
            else:
                v = self.reflex.step(sim.snapshot)
                if v is not None:                          # the edge reflex owns the wheels (robot side)
                    sim.set_wheels(v, v, 300.0)
                else:
                    sim.set_wheels(left * self.wheel_gain, right * self.wheel_gain, 400.0)   # same ramp as Body.apply
            sim.step(1.0 / spec.control_hz)
            self.t += 1
            s = sim.snapshot
            sc = sim.scene_info
            fell = s["pos"][2] < sc.floor_z - 0.03 or (s["upright"] < 0.3 and spec.task != "recover")
            reward, done, info = 0.0, False, {}
            if spec.task != "recover":
                reward -= 0.02 * jerk                          # smooth driving (no deploy-side smoothing to hide flicker)
            if spec.task == "goto":
                dist = math.dist(s["pos"][:2], self.goal)
                reward += 4.0 * (self._prev_dist - dist)           # progress
                self._prev_dist = dist
                reward -= 0.01
                if s["bump"]:
                    reward -= 2.0                              # around things, never through them
                    info["bump"] = True
                    self.bumps += 1
                if dist < 0.06:
                    reward += 3.0
                    done, info["success"] = True, True
            elif spec.task == "desk":
                cell = (int(s["pos"][0] // 0.05), int(s["pos"][1] // 0.05))
                if cell not in self.visited:
                    self.visited.add(cell)
                    reward += 0.1
                fwd = (s["left_mmps"] + s["right_mmps"]) / 2 / spec.max_speed
                reward += 0.02 * max(0.0, fwd)
                if s["bump"]:
                    reward -= 3.0
                    info["bump"] = True
                    self.bumps += 1
                near = float(sim.rays(5, math.radians(90), 0.3).min())     # things on the desk (rays pass over the drop)
                if near < 0.08:
                    reward -= 0.08 * (1 - near / 0.08)          # steer clear early, don't learn "bump, then turn"
                ahead = edge_rays(s, sc.room, 5)[1:4].min()
                if ahead < 0.05:
                    reward -= 0.1 * (1 - ahead / 0.05)         # keep well back from the drop
                info["cells"] = len(self.visited)
            elif spec.task == "wander":
                cell = (int(s["pos"][0] // 0.1), int(s["pos"][1] // 0.1))
                if cell not in self.visited:
                    self.visited.add(cell)
                    reward += 0.1
                fwd = (s["left_mmps"] + s["right_mmps"]) / 2 / spec.max_speed
                reward += 0.02 * max(0.0, fwd)                 # rolling forward beats spinning
                if s["bump"]:
                    reward -= 4.0
                    info["bump"] = True
                    self.bumps += 1
                if float(sim.rays(5, math.radians(90), 0.3).min()) < 0.05:
                    reward -= 0.02                             # don't scrape along things
                info["cells"] = len(self.visited)
            else:   # recover: get upright and stay there
                up = s["upright"]
                reward += 0.5 * (up - self._prev_up) - 0.01
                self._prev_up = up
                self._upright_for = self._upright_for + 1 if up > 0.95 and s["wheel_ground"] else 0
                if self._upright_for >= 10:                  # a full second on its treads
                    reward += 5.0
                    done, info["success"] = True, True
            if s.get("cliff") and spec.task != "recover":
                reward -= 0.3                                  # needing the reflex at all is a near-miss
                info["cliff"] = True
            if fell:
                reward -= 8.0
                done = True
                info["fell"] = True
            truncated = self.t >= spec.episode_s * spec.control_hz
            info["bumps"] = self.bumps
            return self._obs(), float(reward), done, truncated, info
