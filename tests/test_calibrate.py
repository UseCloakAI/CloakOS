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
