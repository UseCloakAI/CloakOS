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
