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


def test_click_to_desk_to_goal_round_trip(tmp_path, monkeypatch):
    """The page maps a click through the corner homography; the goal the driver gets is that point."""
    import numpy as np
    corners = [[0.1, 0.9], [0.9, 0.9], [0.9, 0.1], [0.1, 0.1]]
    size = (0.8, 0.5)
    import cv2
    H = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]))
    click = np.float32([[[0.15, 0.15]]])                  # near the top-left of the image = back-left of the desk
    x, y = cv2.perspectiveTransform(click, H)[0, 0]
    assert x < 0.1 and y > 0.4                            # left, and toward the back (large y)
    from cloakos.behaviors.goto import GoToPolicy
    gp = GoToPolicy(None, None, None)
    gp.set_point(float(x), float(y))
    assert gp.goal == (float(x), float(y)) and gp.frame == "arena"


HIS = [[0.070, 0.315], [0.839, 0.282], [0.881, 0.741], [0.035, 0.763]]   # Weston's desk: front edge at the TOP of the image


def _homography(corners, size):
    import cv2
    import numpy as np
    return cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]))


def _desk_yaw_from_image(H, centre_px, heading_img_rad):
    """What the markers give: map Vector's centre and a point 3 cm-ish ahead of it (image) to the desk."""
    import cv2
    import numpy as np
    p0 = np.float32([[centre_px]])
    p1 = np.float32([[[centre_px[0] + 0.03 * math.cos(heading_img_rad), centre_px[1] + 0.03 * math.sin(heading_img_rad)]]])
    a, b = cv2.perspectiveTransform(p0, H)[0, 0], cv2.perspectiveTransform(p1, H)[0, 0]
    return math.atan2(b[1] - a[1], b[0] - a[0]), (float(a[0]), float(a[1]))


def test_his_desk_end_to_end_turns_toward_a_target_on_the_right(tmp_path, monkeypatch):
    """Weston's corners. A real (non-mirrored) overhead image: physical counter-clockwise, seen from
    above, is counter-clockwise on screen = the image angle atan2(dv, du) DECREASING (v points down)."""
    from cloakos.behaviors.goto import GoToPolicy
    a, _ = _arena(tmp_path, monkeypatch)
    a.save_config({"corners": HIS, "size": [80, 50]})
    assert a.yaw_sign == -1                                   # his desk frame is a mirror image
    H = _homography(HIS, (0.8, 0.5))
    c = (0.45, 0.5)
    # Vector faces North on screen (image up: angle -90°); physical CCW by d -> image angle -90° - d
    y0, pos = _desk_yaw_from_image(H, c, -math.pi / 2)
    y1, _ = _desk_yaw_from_image(H, c, -math.pi / 2 - 0.2)
    dy = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
    assert dy * a.yaw_sign > 0                                # physical CCW moves desk yaw by yaw_sign: the geometry holds
    # a target on the RIGHT of the screen (larger u): its desk point
    import cv2
    import numpy as np
    tgt = cv2.perspectiveTransform(np.float32([[[0.8, 0.5]]]), H)[0, 0]
    gp = GoToPolicy(None, None, None)
    gp.set_point(float(tgt[0]), float(tgt[1]))
    px, py, pyaw = a.to_phys(pos[0], pos[1], y0)
    gx, gy, _ = a.to_phys(float(tgt[0]), float(tgt[1]), 0.0)
    left, right = gp._point_controller({"pos": [px, py, 0.0], "yaw": pyaw}, math.dist(pos, tgt), (gx, gy))
    # facing North with the target to the East (screen right) means turning CLOCKWISE: left wheel forward
    assert left > right and left > 0


def test_disagreement_only_stops_never_flips(tmp_path, monkeypatch):
    a, Brain = _arena(tmp_path, monkeypatch)
    a.save_config({"corners": HIS, "size": [80, 50]})
    import time as _t
    yaw, odo = 0.1, 0.0
    for i in range(60):                                       # camera and odometry turn the SAME way: against -1 geometry
        monkeypatch.setattr(_t, "monotonic", lambda v=3000.0 + i * 0.1: v)
        Brain.world.pose = (0.0, 0.0, odo)
        a.update_pose({"ok": True, "x": 0.4, "y": 0.25, "yaw": yaw, "yaw_src": "markers"})
        yaw += math.radians(5)
        odo += math.radians(5)
    assert a.sign_conflict and a.yaw_sign == -1 and "turn_sign" not in a.config
