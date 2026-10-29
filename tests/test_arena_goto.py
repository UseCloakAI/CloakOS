"""Go to a point on the desk via the Arena (no robot: a fake desk camera and simple kinematics)."""
import asyncio
import math
import time
import types

import numpy as np

from cloakos.arena.session import Arena
from cloakos.behaviors.goto import POINT_ARRIVE_M, GoToPolicy
from cloakos.rl.envs import Spec, goal_features, observe


class FakeBody:
    def __init__(self):
        self.cmd = types.SimpleNamespace(left_mmps=0.0, right_mmps=0.0)
        self.source = "none"

    async def drive(self, l, r, src=None):
        self.cmd.left_mmps, self.cmd.right_mmps = l, r

    async def stop(self):
        await self.drive(0, 0)


class FakeBus:
    def __init__(self):
        self.seen = []

    def emit(self, topic, **kw):
        self.seen.append(topic)


def _setup(tmp_path, x=0.2, y=0.2, yaw=0.0):
    body = FakeBody()
    brain = types.SimpleNamespace(world=types.SimpleNamespace(pose=None, gyro=None, left_mmps=0.0, right_mmps=0.0), body=body)
    a = Arena(brain, tmp_path)
    a.config = {"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "size": [100, 60]}
    a.update_pose({"x": x, "y": y, "yaw": yaw, "ok": True})
    skills = types.SimpleNamespace(has=lambda t: False, snapshot=lambda: {"pos": [0, 0, 0], "yaw": 0.0, "left_mmps": 0, "right_mmps": 0})
    bus = FakeBus()
    gp = GoToPolicy(body, skills, bus)
    gp.arena_src = lambda: a
    return a, gp, body, bus


def test_goal_features_are_what_observe_uses():
    s = {"pos": [0.1, 0.2, 0], "yaw": 0.3, "left_mmps": 0, "right_mmps": 0, "prox_m": None, "cliff": False}
    o = observe(Spec("goto", "robot"), s, (0.5, 0.4), None, np.zeros(2, np.float32))
    assert np.allclose(o[:3], goal_features(s, (0.5, 0.4)))


def test_named_points(tmp_path):
    a, *_ = _setup(tmp_path)
    assert a.named_point("go to the back left corner") == (0.10, 0.50)
    assert a.named_point("front right corner") == (0.90, 0.10)
    assert a.named_point("the middle") == (0.5, 0.3)
    assert a.named_point("the mug") is None
    assert a.in_desk(0.5, 0.3) and not a.in_desk(0.99, 0.3)


def test_drives_to_the_point_accurately(tmp_path):
    a, gp, body, bus = _setup(tmp_path)
    gp.set_point(0.7, 0.4)

    async def go():
        await gp.start()
        x, y, yaw, dt = 0.2, 0.2, 0.0, 0.1
        for _ in range(2000):
            await gp.step(None)
            if gp.status == "arrived":
                return x, y
            l, r = body.cmd.left_mmps / 1000, body.cmd.right_mmps / 1000      # perfect wheels, fresh camera
            v, w = (l + r) / 2, (r - l) / 0.048
            x, y, yaw = x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw + w * dt
            a.update_pose({"x": x, "y": y, "yaw": yaw, "ok": True})
            gp._t0 = gp._best_t = time.time()                  # sim time, not wall time
        return None
    end = asyncio.run(go())
    assert end is not None and math.dist(end, (0.7, 0.4)) < POINT_ARRIVE_M + 0.01
    assert bus.seen == ["goto.arrived"]


def test_holds_still_when_the_camera_is_stale(tmp_path):
    a, gp, body, _ = _setup(tmp_path)
    gp.set_point(0.7, 0.4)
    gp._t0 = gp._best_t = time.time()
    a.pose["t"] = time.time() - 5
    body.cmd.left_mmps = body.cmd.right_mmps = 50
    asyncio.run(gp.step(None))
    assert body.cmd.left_mmps == 0 and body.cmd.right_mmps == 0 and gp.status == "waiting for the desk camera"


def test_teacher_signal_has_the_goal_relative_to_the_robot(tmp_path):
    a, gp, body, _ = _setup(tmp_path, 0.2, 0.3, 0.0)
    gp.set_point(0.6, 0.3)

    async def go():
        await gp.start()
        await gp.step(None)
    asyncio.run(go())
    t = gp.teacher
    assert t["task"] == "goto" and math.isclose(t["goal_rel"][2], 1.0, abs_tol=1e-3)   # straight ahead
    assert t["action"][0] > 0 and t["action"][1] > 0
