"""Arena tracker: desk corners + Vector's marker seen at an angle -> true desk position."""
import math
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cloakos.arena.tracker import ArenaTracker  # noqa: E402


def render_desk(tr, W, H, true, cam_pos, img=(1280, 720), f=950.0):
    w, h = img
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    look = np.array([W / 2, H / 2, 0])
    z = look - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])

    def project(P):
        pc = (R @ (P - cam_pos).T).T
        uv = (K @ pc.T).T
        return uv[:, :2] / uv[:, 2:]

    canvas = np.full((h, w), 180, np.uint8)

    def paste(mid, center, size, yaw=0.0, height=0.0):
        m = cv2.aruco.generateImageMarker(tr.dictionary, mid, 200)
        m = cv2.copyMakeBorder(m, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
        s = size / 2 * 250 / 200
        c, sn = math.cos(yaw), math.sin(yaw)
        loc = np.array([[-s, s], [s, s], [s, -s], [-s, -s]])
        rot = np.array([[c, -sn], [sn, c]]) @ np.array([[0, -1], [1, 0]]).T
        pts = loc @ rot.T + center
        dst = project(np.column_stack([pts, np.full(4, height)])).astype(np.float32)
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [250, 0], [250, 250], [0, 250]]), dst)
        warped = cv2.warpPerspective(m, M, (w, h), borderValue=0)
        mask = cv2.warpPerspective(np.full_like(m, 255), M, (w, h))
        canvas[mask > 0] = warped[mask > 0]

    for i, c in enumerate([(0, 0), (W, 0), (W, H), (0, H)]):
        paste(i, np.array(c), 0.06)
    paste(10, np.array(true[:2]), 0.03, yaw=true[2], height=0.065)
    return canvas


@pytest.mark.parametrize("cam", [(0.4, -0.75, 0.5), (-0.3, -0.5, 0.6)])
def test_tracks_vector_from_any_angle(cam):
    W, H = 0.8, 0.5
    tr = ArenaTracker(W, H, vector_marker=0.03, marker_offset=(0.0, 0.0))
    true = (0.31, 0.22, math.radians(37))
    fix = tr.process(render_desk(tr, W, H, true, np.array(cam)), time.time())
    assert fix.calibrated and fix.vector
    v = fix.vector
    assert math.hypot(v["x"] - true[0], v["y"] - true[1]) < 0.015          # within 1.5 cm
    assert abs((v["yaw"] - true[2] + math.pi) % (2 * math.pi) - math.pi) < math.radians(8)
    assert abs(v["height"] - 0.065) < 0.02
