import asyncio
import types

import numpy as np

from cloakos.calibrate import Calibrator
from cloakos.sim.engine import Sim


def test_map_comparison_against_truth(tmp_path):
    sim = Sim({"sim": {"assets": False, "camera": False}})
    sim.start()
    sim.load("room", 3)

    class FakeMap:
        map = object()
        status = "fake"
        def rays(self, pos, yaw, n, fov, mx):
            t = sim.rays(n, fov, mx)
            t[::3] = mx                                   # a map that misses every third reading
            return t * 1.05                               # and reads 5% far

    link = types.SimpleNamespace(connected=True, target="sim", sim=sim)
    brain = types.SimpleNamespace(link=link, mapper=FakeMap())
    c = Calibrator(brain, tmp_path / "cal.json")
    out = asyncio.run(c._map(12))
    sim.stop()
    assert out["samples"] >= 20 * 11 and out["missed_fraction"] > 0.2
    assert out["range_bias_m"] > 0


def test_camera_lag_from_head_nods(tmp_path):
    from PIL import Image

    world = types.SimpleNamespace(frame=None, frame_id=0, head_angle=0.0)
    pics = [Image.new("L", (80, 45), v) for v in (40, 200)]

    class Body:
        ready = True
        cmd = None

        async def apply(self, a, source=None):
            if a.head_rate:
                self.t, self.up = asyncio.get_running_loop().time(), a.head_rate > 0

        async def stop(self):
            pass

    body = Body()

    async def run():
        async def fake_robot():                   # head sensor moves 0.1 s after a command, the picture 0.35 s after
            while True:
                t = asyncio.get_running_loop().time() - getattr(body, "t", -9)
                up = getattr(body, "up", False)
                world.head_angle = (0.3 if up else 0.0) if t > 0.1 else (0.0 if up else 0.3)
                world.frame = pics[(t > 0.35) == up]
                world.frame_id += 1
                await asyncio.sleep(0.03)

        sim = asyncio.ensure_future(fake_robot())
        await asyncio.sleep(0.05)

        async def set_mode(m):
            pass

        async def set_control(on):
            pass

        brain = types.SimpleNamespace(world=world, body=body, frozen=False, mode="idle", set_mode=set_mode,
                                      link=types.SimpleNamespace(set_control=set_control))
        try:
            return await Calibrator(brain, tmp_path / "c.json")._camera()
        finally:
            sim.cancel()

    out = asyncio.run(run())
    assert out["trials_ok"] >= 4
    assert 0.17 < out["picture_behind_sensors_s"] < 0.35
