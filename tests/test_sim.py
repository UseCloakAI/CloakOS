"""Simulator: physics sanity, and the whole brain running on the sim (no camera: CI has no GL)."""
import asyncio
import math
import time

import pytest

pytest.importorskip("mujoco")

from cloakos.app import Brain  # noqa: E402
from cloakos.config import Config, load_config  # noqa: E402
from cloakos.datalog import Recorder  # noqa: E402
from cloakos.sim import rooms  # noqa: E402
from cloakos.sim.engine import Sim  # noqa: E402

NO_CAM = {"sim": {"camera": False, "assets": False}}


def run_for(sim: Sim, seconds: float) -> None:
    """Step physics directly (no thread) for deterministic tests."""
    import mujoco

    for _ in range(int(seconds / sim._m.opt.timestep)):
        sim._control(sim._m, sim._d, time.monotonic())
        mujoco.mj_step(sim._m, sim._d)
    sim._update_snapshot()


def test_rooms_are_reproducible():
    a, b = rooms.build("room", 123), rooms.build("room", 123)
    assert a.xml == b.xml and a.objects == b.objects
    assert rooms.build("room", 124).xml != a.xml


def test_drives_and_turns_like_vector():
    sim = Sim(NO_CAM)
    sim.load("empty", 1)
    run_for(sim, 0.3)
    p0 = list(sim.snapshot["pos"])
    sim.set_wheels(100, 100)
    run_for(sim, 1.0)
    moved = math.dist(p0[:2], sim.snapshot["pos"][:2])
    assert 0.07 < moved < 0.12                      # ~100 mm/s, after spin-up
    y0 = sim.snapshot["yaw"]
    sim.set_wheels(-50, 50)
    run_for(sim, 0.5)
    turned = abs((sim.snapshot["yaw"] - y0 + math.pi) % (2 * math.pi) - math.pi)
    assert turned > math.radians(15)                # treads spin in place
    assert sim.snapshot["upright"] > 0.95


def test_head_lift_pickup_and_cliff():
    sim = Sim(NO_CAM)
    sim.load("empty", 2)
    sim.set_head(math.radians(40))
    sim.set_lift(None, 3.0)
    run_for(sim, 1.0)
    assert abs(math.degrees(sim.snapshot["head_angle"]) - 40) < 3
    assert sim.snapshot["lift_mm"] > 85
    sim.pick_up(1.0)
    run_for(sim, 0.6)
    assert sim.snapshot["picked_up"]

    sim.load("table", 3)
    run_for(sim, 0.2)
    assert not sim.snapshot["cliff"]
    sim.set_wheels(100, 100)
    for _ in range(160):
        run_for(sim, 0.05)
        if sim.snapshot["cliff"]:
            break
    assert sim.snapshot["cliff"]                    # sees the edge before it goes over
    assert sim.snapshot["pos"][2] > 0.3


def test_balance_world_starts_tipped():
    sim = Sim(NO_CAM)
    sim.load("balance", 4)
    sim._update_snapshot()
    assert sim.snapshot["pitch"] > math.radians(45)


def test_recorder_delete(tmp_path):
    rec = Recorder({"datalog": {"dir": str(tmp_path)}})
    d = rec.start("teleop", 10)
    rec.stop()
    assert rec.delete(d.name) and not d.exists()
    assert not rec.delete("../../etc")


async def test_brain_on_sim_with_freeze(tmp_path):
    cfg = Config(load_config("config.yaml"))
    cfg["datalog"]["dir"] = str(tmp_path)
    cfg["voice"]["enabled"] = False
    cfg["sim"].update(enabled=True, camera=False, assets=False, world="empty", seed=5)
    cfg["robot"]["heartbeat_s"] = 0.2
    brain = Brain(cfg)
    task = asyncio.create_task(brain.run(mode="teleop", web=False))
    for _ in range(80):
        await asyncio.sleep(0.1)
        if brain.link.has_control:
            break
    assert brain.link.has_control and brain.link.target == "sim"
    sim = brain.link.sim

    brain.teleop.input(throttle=1.0)
    for _ in range(10):
        await asyncio.sleep(0.1)
        brain.teleop.input(throttle=1.0)
    assert sim.snapshot["speed"] > 0.05             # teleop drives the sim body

    sim.tap_button(3)                               # triple tap -> freeze
    for _ in range(20):
        await asyncio.sleep(0.1)
        brain.teleop.input(throttle=1.0)
    assert brain.frozen
    await asyncio.sleep(0.5)
    p = list(sim.snapshot["pos"])
    for _ in range(5):
        brain.teleop.input(throttle=1.0)            # driving is ignored while frozen
        await asyncio.sleep(0.1)
    assert math.dist(p[:2], sim.snapshot["pos"][:2]) < 0.01

    sim.tap_button(3)
    for _ in range(20):
        await asyncio.sleep(0.1)
    assert not brain.frozen

    brain.request_stop()
    await asyncio.wait_for(task, 10)
    sim.stop()


def test_camera_thread_falls_back_to_physics_thread(monkeypatch):
    import mujoco
    from cloakos.sim.engine import Sim

    sim = Sim({"sim": {"assets": False}})
    sim.load("empty", 1)
    sim._separate_render = True
    sim.touch(10)

    def broken(*a, **k):
        raise RuntimeError("no GL context on this thread")
    monkeypatch.setattr(mujoco, "Renderer", broken)
    sim._render_once({"thread": True})
    assert sim._separate_render is False and sim.camera_on        # still on, now via the physics thread
    assert "no GL context" in sim.render_error
