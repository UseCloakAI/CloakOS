"""Camera geometry, depth estimation and object detection for the map.

Depth on the real robot: Depth Anything V2 Small (Apache-2.0) gives relative inverse depth;
we turn it into metres using the floor. The camera's height and tilt are known (pose + head
angle), so every floor pixel has a true distance; a robust fit of the model's output against
those gives scale and shift for the whole frame. In the sim, depth is rendered exactly.

Objects on the real robot: RT-DETR (Apache-2.0, COCO classes). In the sim: the simulator's
ground truth (what's in view and unblocked), so the pipeline can be tested end to end.
Both models download from Hugging Face on first use and run on the Mac's GPU (MPS) or CPU.
"""
from __future__ import annotations

import importlib.util
import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

FOVY = math.radians(50.0)        # Vector's camera: ~90 deg horizontal at 16:9
HEAD_PIVOT = np.array([-0.004, 0.0, 0.043])
CAM_IN_HEAD = np.array([0.031, 0.0, 0.0])


@dataclass
class CameraPose:
    pos: np.ndarray        # (3,) world
    R: np.ndarray          # (3,3) columns: forward, left, up (world)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def camera_pose(robot_pos, robot_R: np.ndarray, head_angle: float) -> CameraPose:
    """Robot pose (m, rotation) + head angle (rad, + = up) -> camera pose in the world."""
    Rh = robot_R @ rot_y(-head_angle)          # head pitches about -y (nose up when positive)
    pos = np.asarray(robot_pos, float) + robot_R @ HEAD_PIVOT + Rh @ CAM_IN_HEAD
    return CameraPose(pos, Rh)


def intrinsics(w: int, h: int) -> tuple[float, float, float]:
    f = (h / 2) / math.tan(FOVY / 2)
    return f, w / 2, h / 2


def rays(w: int, h: int) -> np.ndarray:
    """(h, w, 3) camera-frame directions (forward, left, up) with forward = 1."""
    f, cx, cy = intrinsics(w, h)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([np.ones_like(u), -(u - cx) / f, -(v - cy) / f], -1)


def backproject(depth: np.ndarray, cam: CameraPose, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Depth image (metres along the optical axis) -> world points (N,3) + their pixel indices."""
    h, w = depth.shape
    r = rays(w, h)[::stride, ::stride]
    d = depth[::stride, ::stride]
    ok = np.isfinite(d) & (d > 0.02)
    p_cam = r[ok] * d[ok][:, None]
    vv, uu = np.nonzero(ok)
    return cam.pos + p_cam @ cam.R.T, np.stack([vv * stride, uu * stride], 1)


def floor_depth(cam: CameraPose, w: int, h: int, floor_z: float = 0.0) -> np.ndarray:
    """Where each pixel's ray hits the floor plane (metres along the optical axis), NaN if it doesn't."""
    r = rays(w, h) @ cam.R.T
    dz = r[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (floor_z - cam.pos[2]) / dz
    t[(dz > -0.02) | (t <= 0) | (t > 3.0)] = np.nan
    return t


class DepthModel:
    """Depth Anything V2 Small -> metric depth via a floor fit (and the distance sensor)."""

    NAME = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.NAME
        self.is_metric = any(k in self.name.lower() for k in ("metric", "depthpro"))   # outputs metres directly
        self.pipe = None
        self.error: str | None = None

    @staticmethod
    def available() -> bool:
        return all(importlib.util.find_spec(m) for m in ("torch", "transformers"))

    def _load(self) -> None:
        import torch
        from transformers import pipeline

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = pipeline("depth-estimation", model=self.name, device=dev)
        log.info("depth model loaded on %s", dev)

    def relative(self, image) -> np.ndarray:
        if self.pipe is None:
            self._load()
        out = self.pipe(image)
        d = out["predicted_depth"]
        d = d.squeeze().float().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d, np.float32)
        from PIL import Image

        return np.asarray(Image.fromarray(d).resize(image.size, Image.BILINEAR), np.float32)

    def metric(self, image, cam: CameraPose, prox_m: float | None = None, floor_z: float = 0.0) -> np.ndarray | None:
        """Model output -> metres. None if the frame can't be scaled (no floor in view)."""
        rel = self.relative(image)
        h, w = rel.shape
        if self.is_metric:
            return self._metric_direct(rel, cam, prox_m, floor_z)
        fl = floor_depth(cam, w, h, floor_z)
        ys, xs = np.nonzero(np.isfinite(fl[::4, ::4]))
        if len(ys) < 60:
            return None
        d_rel = rel[::4, ::4][ys, xs]
        inv = 1.0 / fl[::4, ::4][ys, xs]
        rng = np.random.default_rng(0)
        best, best_n = None, 0
        for _ in range(60):                           # RANSAC: rel ≈ a * (1/z) + b on floor pixels
            i, j = rng.choice(len(inv), 2, replace=False)
            if abs(inv[i] - inv[j]) < 1e-3:
                continue
            a = (d_rel[i] - d_rel[j]) / (inv[i] - inv[j])
            b = d_rel[i] - a * inv[i]
            if a <= 0:
                continue
            n = int((np.abs(a * inv + b - d_rel) < 0.04 * (np.ptp(d_rel) + 1e-6)).sum())
            if n > best_n:
                best, best_n = (a, b), n
        if best is None or best_n < 30:
            return None
        a, b = best
        with np.errstate(divide="ignore", invalid="ignore"):
            z = a / (rel - b)
        z[(rel - b) <= 1e-6] = np.nan
        if prox_m:                                    # the distance sensor beats the model at the centre
            c = z[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]
            med = np.nanmedian(c)
            if np.isfinite(med) and med > 0 and 0.5 < prox_m / med < 2.0:
                z *= prox_m / med
        return z

    def _metric_direct(self, z: np.ndarray, cam: CameraPose, prox_m: float | None, floor_z: float) -> np.ndarray:
        """A metric model's metres, nudged by the floor (whose true distance the camera geometry gives)
        and the distance sensor. The model was trained on human-height views, so the scale drifts."""
        h, w = z.shape
        z = z.astype(np.float32).copy()
        fl = floor_depth(cam, w, h, floor_z)
        both = np.isfinite(fl) & (z > 0)
        if both.sum() > 200:
            s = float(np.median(fl[both] / z[both]))
            if 0.03 < s < 5.0:                      # at Vector's height these models read ~6x too far
                z *= s
        if prox_m:
            c = z[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]
            med = float(np.nanmedian(c))
            if med > 0 and 0.5 < prox_m / med < 2.0:
                z *= prox_m / med
        return z


class Detector:
    """RT-DETR (COCO) via transformers: boxes + labels."""

    NAME = "PekingU/rtdetr_r18vd_coco_o365"

    def __init__(self, threshold: float = 0.5) -> None:
        self.pipe = None
        self.threshold = threshold

    @staticmethod
    def available() -> bool:
        return all(importlib.util.find_spec(m) for m in ("torch", "transformers"))

    def detect(self, image) -> list[dict]:
        if self.pipe is None:
            import torch
            from transformers import pipeline

            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self.pipe = pipeline("object-detection", model=self.NAME, device=dev)
        out = []
        for d in self.pipe(image, threshold=self.threshold):
            b = d["box"]
            out.append({"label": d["label"], "score": float(d["score"]),
                        "box": (b["xmin"], b["ymin"], b["xmax"], b["ymax"])})
        return out
