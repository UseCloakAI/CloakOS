"""ArenaTracker: where Vector really is on the desk, from any camera that can see it.

Four ArUco markers mark the desk corners (IDs 0-3, measured distances between their centres),
one small marker sits on top of Vector's head (ID 10). The camera can be anywhere that sees
the desk, at any angle: the corners give us the camera's focal length and its exact pose over
the desk, then Vector's marker gives its true position and heading in desk coordinates.

Desk frame: origin at corner 0, x toward corner 1, y toward corner 3, z up, metres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CORNERS = (0, 1, 2, 3)
VECTOR_ID = 10


@dataclass
class Fix:
    """One tracked frame."""
    t: float
    calibrated: bool
    desk_px: list | None = None              # desk outline in the image (4 points)
    H: list | None = None                    # image pixel -> desk plane (3x3), for clicks
    vector: dict | None = None               # x, y (m), yaw (rad), height (m), px (image point)
    seen: list | None = None                 # marker ids seen


def focal_from_homography(H: np.ndarray, w: int, h: int) -> float | None:
    """Single-view focal length from a plane homography (square pixels, centred principal point)."""
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], float)
    Hn = T @ H
    h1, h2 = Hn[:, 0], Hn[:, 1]
    den = h1[2] * h2[2]
    if abs(den) < 1e-12:
        return None
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
    if f2 <= 0:
        den2 = h1[2] ** 2 - h2[2] ** 2
        if abs(den2) < 1e-12:
            return None
        f2 = -((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / den2
    return math.sqrt(f2) if f2 > 0 else None


class ArenaTracker:
    def __init__(self, desk_w: float = 0.8, desk_h: float = 0.5, vector_marker: float = 0.03,
                 marker_offset: tuple[float, float] = (0.009, 0.065)) -> None:
        import cv2

        self.cv2 = cv2
        self.desk_w, self.desk_h = desk_w, desk_h
        self.vector_marker = vector_marker
        self.marker_offset = marker_offset          # marker centre relative to Vector's origin: (forward, up)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(d, p)
        self.dictionary = d
        self.K: np.ndarray | None = None
        self.rvec = self.tvec = None                # desk -> camera
        self._last_corners: dict[int, np.ndarray] = {}

    def desk_points(self) -> np.ndarray:
        w, h = self.desk_w, self.desk_h
        return np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], np.float64)

    def process(self, img: np.ndarray, t: float) -> Fix:
        cv2 = self.cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)} if ids is not None else {}
        for k in CORNERS:                            # corners rarely move: remember the last sighting
            if k in found:
                self._last_corners[k] = found[k]
        fix = Fix(t=t, calibrated=False, seen=sorted(found))
        if all(k in self._last_corners for k in CORNERS):
            self._calibrate(gray.shape[1], gray.shape[0])
            fix.calibrated = self.K is not None
            centres = np.array([self._last_corners[k].mean(0) for k in CORNERS])
            fix.desk_px = centres.round(1).tolist()
            Hinv = cv2.getPerspectiveTransform(centres.astype(np.float32), self.desk_points()[:, :2].astype(np.float32))
            fix.H = Hinv.tolist()
        if fix.calibrated and VECTOR_ID in found:
            fix.vector = self._vector_pose(found[VECTOR_ID])
        return fix

    def _calibrate(self, w: int, h: int) -> None:
        cv2 = self.cv2
        centres = np.array([self._last_corners[k].mean(0) for k in CORNERS], np.float64)
        H, _ = cv2.findHomography(self.desk_points()[:, :2], centres)
        f = focal_from_homography(H, w, h) if H is not None else None
        if f is None or not (0.3 * w < f < 5 * w):
            f = 0.85 * w                              # ~60° horizontal: a typical webcam / phone
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        ok, rvec, tvec = cv2.solvePnP(self.desk_points(), centres, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            self.K, self.rvec, self.tvec = K, rvec, tvec

    def _vector_pose(self, px: np.ndarray) -> dict | None:
        """Vector's marker -> its ground position and heading on the desk."""
        cv2 = self.cv2
        s = self.vector_marker / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)   # ArUco corner order
        ok, rv, tv = cv2.solvePnP(obj, px.astype(np.float64), self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        Rd, _ = cv2.Rodrigues(self.rvec)            # desk -> camera
        Rm, _ = cv2.Rodrigues(rv)                   # marker -> camera
        R = Rd.T @ Rm                                # marker -> desk
        p = Rd.T @ (tv.ravel() - self.tvec.ravel())  # marker centre in desk coords
        fwd = R[:, 1]                                # the marker's "up" edge points along Vector's nose
        yaw = math.atan2(fwd[1], fwd[0])
        dx, dz = self.marker_offset
        x, y = p[0] - dx * math.cos(yaw), p[1] - dx * math.sin(yaw)
        return {"x": round(float(x), 4), "y": round(float(y), 4), "yaw": round(yaw, 4),
                "height": round(float(p[2]), 4), "px": px.mean(0).round(1).tolist()}

    # ── printable sheet ────────────────────────────────────────

    def marker_png(self, marker_id: int, px: int = 600) -> bytes:
        cv2 = self.cv2
        img = cv2.aruco.generateImageMarker(self.dictionary, marker_id, px)
        img = cv2.copyMakeBorder(img, px // 8, px // 8, px // 8, px // 8, cv2.BORDER_CONSTANT, value=255)   # quiet zone
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes()
