"""3D map (voxels, camera geometry, mapping the sim) and the RL environments."""
import asyncio
import math

import numpy as np
import pytest

from cloakos.mapping.vision import CameraPose, backproject, camera_pose, floor_depth, rot_z
from cloakos.mapping.voxels import OCCUPIED, VoxelMap


def test_voxels_mark_hits_and_carve_free_space():
    m = VoxelMap(voxel=0.05, size_xy=4)
    cam = np.array([0.0, 0.0, 0.1])
    wall = np.array([[1.0, y, 0.1] for y in np.linspace(-0.2, 0.2, 9)])
    for _ in range(3):
        m.integrate(cam, wall, np.full((9, 3), 200, np.uint8))
    ijk, col = m.occupied()
    pts = m.center_of(ijk)
    assert len(ijk) and np.allclose(pts[:, 0], 1.0, atol=0.05) and (col == 200).all()
    i, j, k = m.index(np.array([[0.5, 0.0, 0.1]]))[0][0]
    assert m.logodds[i, j, k] < 0                                # seen through = free
    # the wall moves away: carving from the same spot clears it (one hit outweighs four misses)
    for _ in range(40):
        m.integrate(cam, wall + [1.0, 0, 0])
    i, j, k = m.index(wall[4:5])[0][0]
    assert m.logodds[i, j, k] <= OCCUPIED


def test_camera_geometry_round_trip():
    cam = camera_pose((0.2, -0.1, 0.0), rot_z(0.7), math.radians(-10))   # head tipped down
    fl = floor_depth(cam, 64, 36)
    assert np.isfinite(fl[-1]).all()                              # bottom row sees the floor
    pts, _ = backproject(np.nan_to_num(fl, nan=0.0), cam)
    assert np.allclose(pts[:, 2], 0.0, atol=1e-6)                 # floor pixels land on z = 0


def test_voxel_map_save_load(tmp_path):
    m = VoxelMap(voxel=0.05, size_xy=2)
    m.integrate(np.zeros(3), np.array([[0.5, 0, 0.05]] * 3))
    m.save(tmp_path / "m.npz")
    m2 = VoxelMap.load(tmp_path / "m.npz")
    assert (m2.logodds == m.logodds).all() and m2.voxel == m.voxel


def test_mapper_remembers_and_merges(tmp_path):
    from cloakos.bus import Bus
    from cloakos.mapping.mapper import Mapper
    from cloakos.world import World

    class Link:
        connected, target, sim, pose = False, "real", None, None

    mp = Mapper({"datalog": {"dir": str(tmp_path)}}, Link(), World(), Bus())
    mp._use("test", (0, 0))
    a = mp.remember("cup", np.array([1.0, 1.0, 0.0]), 0.9, "camera")
    b = mp.remember("cup", np.array([1.1, 1.0, 0.0]), 0.8, "camera")
    mp.remember("shoe", np.array([-1.0, 0.0, 0.0]), 0.7, "camera")
    assert a is b and a["seen"] == 2 and len(mp.objects) == 2
    assert mp.find("cup") is None                     # two glimpses aren't enough yet
    mp.remember("cup", np.array([1.05, 1.0, 0.0]), 0.8, "camera")
    assert mp.find("cup")["id"] == a["id"] and mp.find("dragon") is None and mp.find("shoe") is None
    mp.save()
    mp2 = Mapper({"datalog": {"dir": str(tmp_path)}}, Link(), World(), Bus())
    mp2._use("test", (0, 0))
    assert {o["label"] for o in mp2.objects} == {"cup", "shoe"}  # remembered across restarts


def test_sim_depth_maps_the_room(tmp_path):
    pytest.importorskip("mujoco")
    from cloakos.mapping.mapper import Mapper
    from cloakos.sim.engine import Sim

    sim = Sim({"sim": {"camera": True, "assets": False, "render_width": 160, "render_height": 90}})
    sim.load("room", 21)
    try:
        import mujoco
        r = mujoco.Renderer(sim._m, 90, 160)
    except Exception:
        pytest.skip("no OpenGL here")
    sim._capture_depth(r, sim._d, sim._m, sim.scene_info)
    f = sim.depth_frame
    assert f["depth"].shape == (90, 160) and np.isfinite(f["depth"]).all()
    mp = Mapper.__new__(Mapper)
    mp.map, mp.objects, mp._dirty = VoxelMap(), [], False
    mp.bus = type("B", (), {"emit": lambda *a, **k: None})()
    mp._integrate(f["depth"], CameraPose(f["pos"], f["R"]), f["rgb"], f["detections"], 2)
    assert mp.map.stats()["occupied"] > 100


gym = pytest.importorskip("gymnasium")


@pytest.mark.parametrize("task,sensors", [("goto", "robot"), ("goto", "rays"), ("explore", "robot"), ("recover", "robot")])
def test_envs_follow_the_gym_api(task, sensors):
    from gymnasium.utils.env_checker import check_env

    from cloakos.rl.envs import VectorEnv

    env = VectorEnv(task, sensors, seed=1)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=2)
    assert obs.shape == env.observation_space.shape
    for _ in range(20):
        obs, r, done, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if done or trunc:
            env.reset()


def test_goto_rewards_progress():
    from cloakos.rl.envs import VectorEnv

    env = VectorEnv("goto", seed=3)
    env.reset(seed=3)
    s = env.sim.snapshot
    ang = math.atan2(env.goal[1] - s["pos"][1], env.goal[0] - s["pos"][0]) - s["yaw"]
    total = 0.0
    for _ in range(120):                    # turn toward the goal, then drive at it
        ang = math.atan2(env.goal[1] - env.sim.snapshot["pos"][1], env.goal[0] - env.sim.snapshot["pos"][0]) - env.sim.snapshot["yaw"]
        turn = max(-1, min(1, 2 * math.sin(ang)))
        _, r, done, _, _ = env.step([0.5 - turn * 0.5, 0.5 + turn * 0.5])
        total += r
        if done:
            break
    assert total > 0


def test_learned_policy_needs_a_model_and_a_goal():
    from cloakos.rl.envs import Spec
    from cloakos.rl.policy import LearnedPolicy

    class Link:
        target, sim = "real", None

    lp = LearnedPolicy(body=None, link=Link())
    assert lp.deployable() == "no policy loaded"
    lp.spec = Spec("goto")
    assert lp.deployable() == "set a goal first"
    lp.goal = (1.0, 0.0)
    assert lp.deployable() is None
    lp.spec = Spec("goto", "rays")
    assert "sim-only" in lp.deployable()


def test_numpy_policy_runs_without_torch(tmp_path):
    from cloakos.rl.policy import NumpyPolicy

    rng = np.random.default_rng(0)
    np.savez(tmp_path / "policy.npz", w0=rng.normal(size=(8, 9)), b0=np.zeros(8), w1=rng.normal(size=(2, 8)) * 5, b1=np.zeros(2))
    p = NumpyPolicy(tmp_path / "policy.npz")
    a, _ = p.predict(rng.normal(size=9))
    assert a.shape == (2,) and (np.abs(a) <= 1).all()


def test_exported_policy_matches_sb3(tmp_path):
    pytest.importorskip("stable_baselines3")
    from pathlib import Path

    from stable_baselines3 import PPO

    from cloakos.rl.envs import VectorEnv
    from cloakos.rl.policy import NumpyPolicy
    from cloakos.rl.train import export_numpy

    model = PPO("MlpPolicy", VectorEnv("goto", seed=0), n_steps=64, batch_size=32, device="cpu",
                policy_kwargs={"net_arch": [32, 32]})
    export_numpy(model, Path(tmp_path / "policy.npz"))
    np_pol = NumpyPolicy(tmp_path / "policy.npz")
    n = model.observation_space.shape[0]
    for o in np.random.default_rng(1).normal(size=(20, n)).astype(np.float32):
        assert np.allclose(model.predict(o, deterministic=True)[0], np_pol.predict(o)[0], atol=1e-5)
