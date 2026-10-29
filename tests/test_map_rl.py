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


@pytest.mark.parametrize("task,sensors", [("goto", "robot"), ("goto", "rays"), ("explore", "robot"), ("recover", "robot"), ("desk", "map")])
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
    lp.spec = Spec("desk", "map")                 # the desk teacher sees exact edges: sim only
    assert "only runs in the sim" in lp.deployable()


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


def test_desk_sizes_follow_your_desk(monkeypatch):
    from cloakos.sim.rooms import build, desk_size

    monkeypatch.setenv("CLOAK_DESK", "90x60")
    rng = np.random.default_rng(0)
    sizes = [desk_size(rng) for _ in range(200)]
    mine = [s for s in sizes if abs(s[0] - 0.45) < 0.05 and abs(s[1] - 0.3) < 0.035]
    assert 60 < len(mine) < 160 and all(0.2 <= w <= 0.6 for w, _ in sizes if (w, _) not in mine)
    sc = build("desk", 3)
    assert sc.floor_z > 0.5 and len(sc.obstacles) <= 4


def test_pixel_student_matches_torch_and_reads_recordings(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from cloakos.rl.pixelnet import PixelNet
    from cloakos.rl.student import build, fold, load_real

    net = build(torch).eval()
    np.savez(tmp_path / "p.npz", **fold(net))
    f, p = np.random.randint(0, 255, (2, 3, 54, 96), dtype=np.uint8), np.random.rand(7).astype(np.float32)
    with torch.no_grad():
        want = net(torch.from_numpy(np.concatenate([f[0], f[1]])[None]).float() / 255, torch.from_numpy(p[None]))[0].numpy()
    assert np.allclose(PixelNet(tmp_path / "p.npz").act(f[0], f[1], p), want, atol=1e-4)

    rec = tmp_path / "ep"
    (rec / "frames").mkdir(parents=True)
    lines = []
    for i in range(4):
        Image.new("RGB", (320, 180), (i * 40, 0, 0)).save(rec / f"frames/{i}.jpg")
        lines.append({"i": i, "obs": {"left_mmps": 50, "right_mmps": 40, "cliff": False}, "frame": f"frames/{i}.jpg",
                      "action": {"left_mmps": 75, "right_mmps": -75}, "action_source": "teleop" if i != 2 else "wander"})
    (rec / "steps.jsonl").write_text("\n".join(__import__("json").dumps(x) for x in lines))
    part = load_real([str(rec)])
    assert part["frames"].shape == (3, 3, 54, 96) and np.allclose(part["labels"][0], [0.5, -0.5])
    assert list(part["prev"]) == [0, 0, 2]          # the frame after a gap has no "before"


def test_free_spot_too_big_for_desk():
    import numpy as np
    from cloakos.sim.rooms import _free_spot
    assert _free_spot(np.random.default_rng(0), (0.05, 0.05), [], 0.04, margin=0.04) is None


def test_edge_reflex_matches_robot():
    from cloakos.rl.envs import EdgeReflex
    r = EdgeReflex(delay_steps=2, hz=10)
    s = {"cliff": False, "left_mmps": 150.0, "right_mmps": 150.0}
    assert r.step(s) is None
    s = {**s, "cliff": True}
    assert r.step(s) is None                     # Wi-Fi: the brain hears about it a little later
    out = [r.step(s) for _ in range(15)]
    assert out[0] is None and out[1] == -75.0    # was driving forward: reverses, at the robot's 75 mm/s
    assert out.count(-75.0) >= 13                # 1.3 s at full speed in, then tries again while still on the edge
    s2 = {**s, "cliff": False}
    tail = [r.step(s2) for _ in range(30)]
    assert 0.0 in tail and tail[-1] is None      # stops once clear, then gives the wheels back


def test_guided_plan_and_keep_training(tmp_path):
    import json
    import time

    from cloakos.rl.manager import Trainer

    t = Trainer(tmp_path)
    started = []
    t.start = lambda **kw: started.append(kw) or f"run{len(started)}"
    t.running = lambda: None
    assert t.plan()["stage"] == 1 and t.plan()["start"]["task"] == "desk"      # nothing yet: the simulator first
    d = t.root / "desk-map-1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"task": "desk", "sensors": "map", "created": 1}))
    (d / "policy.npz").write_bytes(b"x")
    (d / "exam.json").write_text(json.dumps({"passed": True, "score": 1050}))
    p = t.plan([{"name": "ep1", "frames": 50, "created": time.time()}])
    assert p["stage"] == 2 and p["start"]["teacher"] == "desk-map-1" and p["start"]["real"] == ["ep1"]
    t.set_auto(True)
    assert t.auto_tick([]) == "run1" and started[-1]["hours"] == t.SESSION_H
    # the Mac dies mid-session: the session's folder has a checkpoint but no "finished"
    run = t.root / "run1"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"task": "desk", "sensors": "map", "hours": 2, "trained_s": 1800}))
    (run / "model.zip").write_bytes(b"x")
    resumed = []
    t.resume_run = lambda name: resumed.append(name) or name
    t2 = Trainer(tmp_path)                             # CloakOS starts again after the reboot
    t2.start, t2.running, t2.resume_run = t.start, t.running, t.resume_run
    assert t2.auto["on"] and t2.auto_tick([]) == "run1" and resumed == ["run1"]
    t2.set_auto(False)
    assert t2.auto_tick([]) is None and not json.loads((tmp_path / "keep_training.json").read_text())["on"]
