import math

from cloakos.arena.heading import HeadingGate


def test_still_robot_jumps_are_rejected():
    g = HeadingGate()
    for _ in range(3):
        y, ok = g.update(0.5, "light")
    assert ok and abs(y - 0.5) < 1e-6
    for bad in (math.radians(138), math.radians(-174), math.radians(-42)):   # what the live samples did
        y, ok = g.update(bad, "light", None)
        assert not ok and abs(y - 0.5) < 1e-6


def test_not_trusted_until_consistent_then_relocks():
    g = HeadingGate()
    assert g.update(1.0, "markers")[1] is False
    g.update(2.5, "markers")                  # resets the candidate run, then holds steady
    y, ok = g.update(2.5, "markers")
    assert not ok
    y, ok = g.update(2.5, "markers")
    assert ok and abs(y - 2.5) < 1e-6


def test_odometry_turn_is_followed():
    g = HeadingGate()
    for _ in range(3):
        g.update(0.0, "light", odo=0.0)
    y, ok = g.update(1.0, "light", odo=1.0)    # Vector says it turned 1 rad: a 1 rad change is fine
    assert ok and abs(y - 1.0) < 1e-6


def test_still_robot_ignores_steady_90_degree_flip_and_set_pins_it():
    g = HeadingGate()
    g.set(math.pi / 2, odo=0.0)
    for _ in range(20):                       # the glare pairing says "east" for 2 s, Vector never turned
        y, ok = g.update(0.0, "markers", odo=0.0)
        assert not ok and abs(y - math.pi / 2) < 1e-6


def _arena(tmp_path, monkeypatch):
    from cloakos.arena.session import Arena

    monkeypatch.setenv("CLOAK_ENV_FILE", str(tmp_path / ".env"))

    class World:
        pose, gyro = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    class Brain:
        world = World()

    return Arena(Brain(), tmp_path), Brain


def test_one_heading_page_and_server_agree(tmp_path, monkeypatch):
    a, _ = _arena(tmp_path, monkeypatch)
    for _ in range(3):                           # page says 88° from the markers: the server reports that same 88°
        r = a.update_pose({"ok": True, "x": 0.4, "y": 0.25, "yaw": math.radians(88), "yaw_src": "markers"})
    assert abs(a.heading() - math.radians(88)) < 1e-6
    assert abs(a.imu()["yaw"] - math.radians(88)) < 1e-6


def test_flip_is_taken_at_once(tmp_path, monkeypatch):
    a, _ = _arena(tmp_path, monkeypatch)
    for _ in range(3):
        a.update_pose({"ok": True, "x": 0.4, "y": 0.25, "yaw": 0.0, "yaw_src": "markers"})
    a.set_heading(math.pi)                       # Flip heading / "which way it faces"
    assert abs(abs(a.heading()) - math.pi) < 1e-6
    a.update_pose({"ok": True, "x": 0.4, "y": 0.25, "yaw": math.pi, "yaw_src": "markers"})
    assert a.pose["yaw_src"] == "markers" and abs(abs(a.heading()) - math.pi) < 1e-6


def test_motion_check_flips_a_backwards_camera():
    from cloakos.arena.heading import MotionCheck
    m = MotionCheck()
    assert m.update(0.30, 0.20, True, -math.pi / 2)[0] is None       # anchor
    assert m.update(0.32, 0.20, True, -math.pi / 2)[0] is None       # only 2 cm
    v, mdir = m.update(0.335, 0.20, True, math.pi)                   # drove east 3.5 cm, camera said west
    assert v == "flip" and abs(mdir) < 1e-6
    assert m.update(0.40, 0.20, False, 0.0) == (None, None)          # not driving forward: no verdict
    m.update(0.40, 0.20, True, 0.0)
    assert m.update(0.44, 0.20, True, 0.05)[0] == "ok"


def test_arena_flips_and_confirms_from_driving(tmp_path, monkeypatch):
    a, Brain = _arena(tmp_path, monkeypatch)
    Brain.world.left_mmps = Brain.world.right_mmps = 0.0
    for _ in range(3):                           # camera locks onto the wrong end: facing west
        a.update_pose({"ok": True, "x": 0.30, "y": 0.2, "yaw": math.pi, "yaw_src": "markers"})
    assert not a.heading_confirmed
    Brain.world.left_mmps = Brain.world.right_mmps = 40.0
    for i in range(6):                           # but it drives east
        a.update_pose({"ok": True, "x": 0.30 + 0.01 * i, "y": 0.2, "yaw": math.pi, "yaw_src": "markers"})
    assert a.heading_confirmed and a.config.get("flip") is True
    assert abs(a.heading()) < 0.05


def test_camera_lag_is_measured_from_a_turn():
    from cloakos.arena.lag import LagEstimator
    e, lag = LagEstimator(), 0.24
    for i in range(80):                          # turning at 1 rad/s, odometry at 50 Hz, camera at 10 Hz
        t = i * 0.02
        e.add_odo(t, math.sin(t * 3) * 1.5)
        if i % 5 == 0 and t > lag:
            e.add_cam(t, math.sin((t - lag) * 3) * 1.5)
    assert abs(e.estimate() - lag) < 0.05
