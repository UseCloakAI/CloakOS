"""Arena ranges: the overhead camera's stand-in for the sim's rays (no robot needed)."""
import math
import time
import types

import numpy as np

from cloakos.arena.ranges import EDGE_MARGIN as M, NOSE, desk_rays
from cloakos.rl.envs import RAY_FOV, RAY_MAX, edge_rays


def test_straight_at_an_edge():
    r = desk_rays(0.5, 0.3, 0.0, (1.0, 0.6), 3, RAY_FOV, RAY_MAX)
    assert math.isclose(r[1], 0.5 - NOSE - M, abs_tol=1e-5)          # middle ray: the right-hand edge
    assert math.isclose(r[0], 0.3 - M, abs_tol=1e-5) and math.isclose(r[2], 0.3 - M, abs_tol=1e-5)   # +-90 deg: front/back edges


def test_matches_the_sims_edge_rays():
    # the sim's desk is centred on 0; the Arena's has its origin at a corner: same answer once shifted
    for x, y, yaw in [(0.2, 0.1, 0.4), (-0.3, 0.05, 2.5), (0.0, -0.2, -1.2)]:
        sim = edge_rays({"pos": [x, y, 0], "yaw": yaw}, (0.5, 0.3), 9)
        ar = desk_rays(x + 0.5, y + 0.3, yaw, (1.0, 0.6), 9, RAY_FOV, RAY_MAX)
        assert np.allclose(sim, ar, atol=1e-4)


def test_capped_and_obstacles():
    big = desk_rays(2.0, 2.0, 0.0, (4.0, 4.0), 5, RAY_FOV, RAY_MAX)
    assert np.allclose(big, RAY_MAX - M)
    r = desk_rays(2.0, 2.0, 0.0, (4.0, 4.0), 5, RAY_FOV, RAY_MAX, [(2.5, 2.0, 0.05)])
    assert math.isclose(r[2], 0.5 - NOSE - 0.05, abs_tol=1e-5) and math.isclose(r[0], RAY_MAX - M, abs_tol=1e-5)


def _arena(tmp_path, pose):
    from cloakos.arena.session import Arena
    brain = types.SimpleNamespace(world=types.SimpleNamespace(pose=None, gyro=None))
    a = Arena(brain, tmp_path)
    a.config = {"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "size": [100, 60]}
    if pose:
        a.update_pose({**pose, "ok": True})
    return a


def test_arena_session_ranges_and_status(tmp_path):
    a = _arena(tmp_path, None)
    assert a.ranges_status() == "Arena isn't tracking Vector" and a.ranges(9, RAY_FOV, RAY_MAX) is None
    a = _arena(tmp_path, {"x": 0.5, "y": 0.3, "yaw": 0.0})
    assert a.ranges_status() is None
    assert np.allclose(a.ranges(3, RAY_FOV, RAY_MAX), [0.3 - M, 0.5 - NOSE - M - 0.02, 0.3 - M], atol=1e-5)
    a.pose["t"] = time.time() - 10                                # lost
    assert a.ranges(3, RAY_FOV, RAY_MAX) is None


def test_predict_pose_and_stop_buffer():
    from cloakos.arena.ranges import COMMAND_TO_MOTION_S, STOP_MARGIN, STOP_S, buffered, predict_pose, stop_buffer
    x, y, yaw = predict_pose(0.0, 0.0, 0.0, 0.1, 0.1, 0.5)
    assert math.isclose(x, 0.05, abs_tol=1e-9) and y == 0 and yaw == 0
    x, y, yaw = predict_pose(0.0, 0.0, 0.0, -0.024, 0.024, 1.0)        # spin in place: 1 rad/s
    assert math.isclose(yaw, 1.0, abs_tol=1e-9) and abs(x) < 1e-9 and abs(y) < 1e-9
    assert math.isclose(stop_buffer(0.1), 0.1 * (COMMAND_TO_MOTION_S + STOP_S) + STOP_MARGIN)
    b = buffered(np.full(3, 0.5, np.float32), RAY_FOV, 0.1)
    assert math.isclose(b[1], 0.5 - stop_buffer(0.1), abs_tol=1e-6) and math.isclose(b[0], 0.5, abs_tol=1e-6)


def test_ranges_priority_and_explicit_none(tmp_path):
    from cloakos.rl.policy import RANGES_STATE, ranges
    link = types.SimpleNamespace(target="robot", sim=None)
    s = {"pos": [0, 0, 0], "yaw": 0.0}
    assert ranges(link, None, s, 3) is None and RANGES_STATE["source"] is None and RANGES_STATE["why"]
    a = _arena(tmp_path, {"x": 0.5, "y": 0.3, "yaw": 0.0})
    assert ranges(link, None, s, 3, a) is not None and RANGES_STATE["source"] == "arena"


def test_student_learns_from_teacher_steps(tmp_path):
    import json
    from PIL import Image
    from cloakos.rl.student import load_real
    (tmp_path / "frames").mkdir()
    Image.new("RGB", (320, 180)).save(tmp_path / "frames" / "0.jpg")
    rows = [{"i": 0, "obs": {}, "action": {"left_mmps": 0, "right_mmps": 0}, "action_source": "learned",
             "teacher": {"action": [0.5, -0.5]}, "arena": {"ranges": [0.3] * 9}, "frame": "frames/0.jpg"},
            {"i": 1, "obs": {}, "action": {}, "action_source": "learned", "frame": "frames/0.jpg"}]   # no teacher: skipped
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    part = load_real([str(tmp_path)])
    assert part is not None and len(part["labels"]) == 1 and np.allclose(part["labels"][0], [0.5, -0.5])
