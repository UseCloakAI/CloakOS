"""Ranges from the overhead camera: what the sim's rays() would read, worked out from the Arena.

The Arena knows where Vector is on the desk (x, y, heading, desk frame: origin at the front-left
corner, metres) and how big the desk is. A desk's drop-off reads as a wall, the same as the sim's
edge_rays, and anything else the Arena knows about (today only the charger) is a round obstacle.
Same fan as training: n rays across fov, centred on the nose, from 4 cm in front of the centre.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

NOSE = 0.04            # the ray origin sits this far ahead of the centre, like the sim's prox site
EDGE_MARGIN = 0.03     # edges read this much nearer, the same margin as the sim's edge_rays


def _hit_circle(x0: float, y0: float, dx: float, dy: float, cx: float, cy: float, r: float) -> float:
    """Distance along a unit ray to a circle, inf if it misses (0 if the nose is inside it)."""
    fx, fy = x0 - cx, y0 - cy
    c = fx * fx + fy * fy - r * r
    if c <= 0:
        return 0.0
    b = fx * dx + fy * dy
    disc = b * b - c
    if b > 0 or disc < 0:
        return math.inf
    return -b - math.sqrt(disc)


def desk_rays(x: float, y: float, yaw: float, size: tuple[float, float], n: int, fov: float, max_range: float,
              obstacles: Iterable[tuple[float, float, float]] = ()) -> np.ndarray:
    """Range (m) to the desk edge or an obstacle (x, y, radius) along each ray, capped at max_range."""
    w, d = size
    x0, y0 = x + NOSE * math.cos(yaw), y + NOSE * math.sin(yaw)
    obs = list(obstacles)
    out = np.full(n, max_range, np.float32)
    for i, a in enumerate(np.linspace(-fov / 2, fov / 2, n)):
        dx, dy = math.cos(yaw + a), math.sin(yaw + a)
        t = max_range
        if dx > 1e-6: t = min(t, (w - x0) / dx)
        elif dx < -1e-6: t = min(t, -x0 / dx)
        if dy > 1e-6: t = min(t, (d - y0) / dy)
        elif dy < -1e-6: t = min(t, -y0 / dy)
        t -= EDGE_MARGIN                                  # like edge_rays, even at full range
        for cx, cy, r in obs:
            t = min(t, _hit_circle(x0, y0, dx, dy, cx, cy, r))
        out[i] = max(0.0, t)
    return out


# Measured on the user's robot (data/calibration.json, Oct 28): Wi-Fi control is always late.
COMMAND_TO_MOTION_S = 0.41
STOP_S = 0.446
WHEEL_GAIN = 0.86
WHEEL_BASE = 0.048     # m between Vector's treads
STOP_MARGIN = 0.02     # m on top of the stopping distance


def predict_pose(x: float, y: float, yaw: float, left_mps: float, right_mps: float, dt: float) -> tuple[float, float, float]:
    """Where Vector will be dt seconds from now at these wheel speeds (differential drive, exact arc)."""
    v, w = (left_mps + right_mps) / 2, (right_mps - left_mps) / WHEEL_BASE
    if abs(w) < 1e-6:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    y2 = yaw + w * dt
    return x + v / w * (math.sin(y2) - math.sin(yaw)), y - v / w * (math.cos(y2) - math.cos(yaw)), math.atan2(math.sin(y2), math.cos(y2))


def stop_buffer(speed_mps: float) -> float:
    """How far Vector goes before a stop takes hold: speed x (command lag + stopping time) + a margin."""
    return abs(speed_mps) * (COMMAND_TO_MOTION_S + STOP_S) + STOP_MARGIN


def buffered(rays: np.ndarray, fov: float, speed_mps: float) -> np.ndarray:
    """Rays shortened by the stopping distance in the direction of travel (full ahead, none sideways)."""
    n = len(rays)
    ang = np.linspace(-fov / 2, fov / 2, n)
    along = np.maximum(0.0, np.cos(ang)) if speed_mps >= 0 else np.zeros(n)
    return np.maximum(0.0, rays - stop_buffer(speed_mps) * along).astype(np.float32)
