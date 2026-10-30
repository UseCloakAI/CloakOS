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
