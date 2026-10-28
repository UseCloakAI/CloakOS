"""PixelNet: the desk student. Camera pixels (this frame + the one before) and Vector's own body
sensors in, wheel speeds out. Trained by imitating the desk teacher (cloakos.rl.student); runs in numpy."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..mapping.visionnet import _conv

IMG_W, IMG_H = 96, 54
PROPRIO = 7          # wheel speeds, distance sensor, cliff, upright, last action (same as observe(desk, robot))


def prepare(img) -> np.ndarray:
    """PIL image or HxWx3 uint8 array -> (3, 54, 96) uint8."""
    from PIL import Image

    a = np.asarray(img)
    if a.shape != (IMG_H, IMG_W, 3):
        im = img if hasattr(img, "resize") else Image.fromarray(a.astype(np.uint8))
        a = np.asarray(im.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR))
    return np.ascontiguousarray(a.transpose(2, 0, 1))


class PixelNet:
    def __init__(self, path: str | Path) -> None:
        z = np.load(path)
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n_conv"]))]
        self.fc = [(z[f"fw{i}"], z[f"fb{i}"]) for i in range(3)]

    def act(self, frame: np.ndarray, prev: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        """frame, prev: (3, 54, 96) uint8; proprio: (7,) -> wheel actions in -1..1."""
        x = np.concatenate([frame, prev]).astype(np.float32) / 255.0 - 0.5
        for w, b in self.layers:
            x = np.maximum(_conv(x, w, b, 2, 2), 0.0)
        x = np.concatenate([x.reshape(-1), np.asarray(proprio, np.float32)])
        x = np.maximum(self.fc[0][0] @ x + self.fc[0][1], 0.0)
        x = np.maximum(self.fc[1][0] @ x + self.fc[1][1], 0.0)
        return np.tanh(self.fc[2][0] @ x + self.fc[2][1]).astype(np.float32)
