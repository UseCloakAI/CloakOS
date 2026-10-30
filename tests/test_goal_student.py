import math

import numpy as np
import pytest

from cloakos.rl.goal_student import BODY, PROP, DeadReckon, real_goal_steps, scripted_teacher


def test_goal_from_odometry_matches_the_overhead_goal():
    # Arena: Vector at (0.2, 0.1) facing +y (90°); goal 0.3 m straight ahead of it at (0.2, 0.4).
    # Vector's own odometry has a different origin and heading: at (1.0, 2.0) facing 0.
    steps = [{"teacher": {"task": "goto", "goal": [0.2, 0.4], "action": [0.3, 0.3]},
              "arena": {"ok": True, "x": 0.2, "y": 0.1, "yaw": math.pi / 2},
              "obs": {"pose": [1000.0, 2000.0, 0.0]}},
             # it drove 0.1 m forward (odometry +x): the goal is now 0.2 m ahead, even with no overhead fix
             {"teacher": {"task": "goto", "goal": [0.2, 0.4], "action": [0.3, 0.3]},
              "arena": {"ok": False}, "obs": {"pose": [1100.0, 2000.0, 0.0]}}]
    out = real_goal_steps(steps)
    assert len(out) == 2
    d0, s0, c0 = out[0][1]
    d1, s1, c1 = out[1][1]
    assert abs(d0 * 3 - 0.3) < 1e-6 and abs(s0) < 1e-6 and abs(c0 - 1) < 1e-6
    assert abs(d1 * 3 - 0.2) < 1e-6 and abs(s1) < 1e-6
    assert out[0][2].tolist() == [pytest.approx(0.3), pytest.approx(0.3)]


def test_dead_reckoning_drifts_only_a_little():
    rng = np.random.default_rng(0)
    o = DeadReckon(rng, (0.0, 0.0, 0.0))
    for _ in range(50):                               # 5 s straight at 100 mm/s
        s = o.step(100.0, 100.0, 0.1)
    assert 0.45 < s["pos"][0] < 0.55 and abs(s["pos"][1]) < 0.05


def test_teacher_turns_toward_the_goal_and_never_reverses():
    s = {"pos": [0.0, 0.0, 0.0], "yaw": 0.0}
    left = scripted_teacher(s, (0.0, 0.5), np.ones(11))          # goal on the left: turn left (right wheel faster)
    assert left[1] > left[0]
    ahead = scripted_teacher(s, (0.5, 0.0), np.ones(11))
    assert ahead[0] > 0 and ahead[1] > 0
    blocked = scripted_teacher(s, (0.5, 0.0), np.full(11, 0.03))  # an edge right ahead: turn, don't drive into it
    assert blocked[0] + blocked[1] == pytest.approx(0.0)


def test_network_shapes_and_one_train_step():
    torch = pytest.importorskip("torch")
    from cloakos.rl.goal_student import build, fold, train_round
    from cloakos.rl.pixelnet import IMG_H, IMG_W, PixelNet
    net = build(torch)
    rng = np.random.default_rng(0)
    n = 8
    part = {"frames": rng.integers(0, 255, (n, 3, IMG_H, IMG_W), dtype=np.uint8), "prev": np.arange(n),
            "proprio": rng.normal(size=(n, PROP)).astype(np.float32), "labels": rng.uniform(-1, 1, (n, 2)).astype(np.float32)}
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = train_round(torch, net, opt, [(part, 1)], rng, epochs=1, size=4)
    assert len(losses) == 2 and all(np.isfinite(losses))
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "p.npz"), **fold(net))
        a = PixelNet(os.path.join(d, "p.npz")).act(part["frames"][0], part["frames"][1], part["proprio"][0])
    assert a.shape == (2,) and np.all(np.abs(a) <= 1)
    assert PROP == BODY + 3
