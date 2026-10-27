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
    "recover": "Get back on its treads after a fall (experimental: train long)",
}
SENSORS = {"map": "Map ranges + robot sensors (recommended)", "robot": "Robot sensors only"}
ALIASES = {"explore": "wander", "rays": "map"}       # names used by older policies
RAY_FOV = math.radians(180)
RAY_MAX = 1.0


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
        base = {"goto": 9, "wander": 7, "recover": 12}[self.task]
        return base + (self.n_rays if self.sensors == "map" else 0)


def make_spec(task: str, sensors: str = "map") -> Spec:
    task, sensors = ALIASES.get(task, task), ALIASES.get(sensors, sensors)
    if task == "recover":
        return Spec("recover", "robot", control_hz=10.0, max_speed=200.0, episode_s=10.0)
    return Spec(task, sensors if sensors in SENSORS else "map", episode_s=40.0 if task == "wander" else 30.0)


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
    elif spec.task == "wander":
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
                     worlds_per_reload: int = 8) -> None:
            super().__init__()
            self.spec_ = make_spec(task, sensors)
            n = self.spec_.obs_size
            self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (3 if task == "recover" else 2,), np.float32)
            self.sim = Sim({"sim": {"camera": False, "assets": False}})
            self.rng = np.random.default_rng(seed)
            self.reload_every = worlds_per_reload
            self.episodes = 0
            self.goal: tuple[float, float] | None = None
            self.last_action = np.zeros(2, np.float32)
            self.t = 0
            self.visited: set[tuple[int, int]] = set()

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
            r = r * self.rng.normal(1.0, 0.05, r.shape)
            r[self.rng.random(r.shape) < 0.12] = RAY_MAX
            return np.clip(r, 0.0, RAY_MAX).astype(np.float32)

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
                world = "empty" if task == "recover" else str(self.rng.choice(["room", "room", "empty", "table"]))
                self.sim.load(world, int(self.rng.integers(1, 1_000_000)))
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
            self.last_action = np.zeros(2, np.float32)
            self.t = 0
            self.bumps = 0
            self.visited = set()
            # latency like the real robot over Wi-Fi: commands land 0-200 ms late, readings 0-100 ms stale
            from collections import deque
            self.act_delay = int(self.rng.integers(0, 3))
            self.obs_delay = int(self.rng.integers(0, 2))
            self._act_q = deque([np.zeros(3, np.float32)] * self.act_delay)
            self._obs_q = deque(maxlen=self.obs_delay + 1)
            s = self.sim.snapshot
            self._prev_dist = math.dist(s["pos"][:2], self.goal) if self.goal else 0.0
            self._prev_up, self._upright_for = s["upright"], 0
            return self._obs(), {}

        def step(self, action):
            spec, sim = self.spec_, self.sim
            a = np.clip(np.asarray(action, np.float32), -1, 1)
            self._act_q.append(np.pad(a, (0, 3 - len(a))))
            a = self._act_q.popleft()[:len(a)]          # what actually reaches the robot this step
            if spec.task == "recover":
                sim.set_lift(None, float(a[2]) * 3.0)
                left, right = float(a[0]) * spec.max_speed, float(a[1]) * spec.max_speed
                self.last_action = a[:2].astype(np.float32)
            else:
                left, right = float(a[0]) * spec.max_speed, float(a[1]) * spec.max_speed
                self.last_action = a.astype(np.float32)
            sim.set_wheels(left, right, 400.0 if spec.task != "recover" else 0.0)   # same ramp as Body.apply
            sim.step(1.0 / spec.control_hz)
            self.t += 1
            s = sim.snapshot
            sc = sim.scene_info
            fell = s["pos"][2] < sc.floor_z - 0.03 or (s["upright"] < 0.3 and spec.task != "recover")
            reward, done, info = 0.0, False, {}
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
            elif spec.task == "wander":
                cell = (int(s["pos"][0] // 0.1), int(s["pos"][1] // 0.1))
                if cell not in self.visited:
                    self.visited.add(cell)
                    reward += 0.1
                fwd = (s["left_mmps"] + s["right_mmps"]) / 2 / spec.max_speed
                reward += 0.02 * max(0.0, fwd)                 # rolling forward beats spinning
                if s["bump"]:
                    reward -= 1.0
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
                fw = (s["left_mmps"] + s["right_mmps"]) / 2 / spec.max_speed
                reward += 0.4 * max(0.0, -fw) - 0.15           # edge sensors fired: back off, don't sit and push
                info["cliff"] = True
            if fell:
                reward -= 8.0
                done = True
                info["fell"] = True
            truncated = self.t >= spec.episode_s * spec.control_hz
            info["bumps"] = self.bumps
            return self._obs(), float(reward), done, truncated, info
