"""VisionNet: a tiny camera-only network that reads Vector's frame and answers "how far is the nearest
thing" in 11 directions (obstacles and table edges). Trained on simulator pictures; runs in numpy."""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMG_W, IMG_H = 96, 54


def _conv(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """x: (C, H, W), w: (O, C, k, k)."""
    o, c, k, _ = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    h = (x.shape[1] - k) // stride + 1
    wd = (x.shape[2] - k) // stride + 1
    s0, s1, s2 = x.strides
    cols = np.lib.stride_tricks.as_strided(x, (c, k, k, h, wd), (s0, s1, s2, s1 * stride, s2 * stride))
    out = w.reshape(o, -1) @ cols.reshape(c * k * k, h * wd)
    return (out + b[:, None]).reshape(o, h, wd)


class VisionNet:
    def __init__(self, path: str | Path) -> None:
        z = np.load(path)
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n_conv"]))]
        self.fc = [(z["fw0"], z["fb0"]), (z["fw1"], z["fb1"])]
        self.meta = {k: z[k].tolist() for k in z.files if k.startswith("m_")}

    @staticmethod
    def prepare(img) -> np.ndarray:
        """PIL image or HxWx3 uint8 array -> (3, 54, 96) float32 in 0..1."""
        from PIL import Image

        im = img if hasattr(img, "resize") else Image.fromarray(np.asarray(img, np.uint8))
        im = im.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
        return (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)

    def predict(self, img) -> np.ndarray:
        """11 ranges in metres (0..1.0), left to right like the map's rays."""
        x = img if isinstance(img, np.ndarray) and img.shape == (3, IMG_H, IMG_W) and img.dtype == np.float32 else self.prepare(img)
        x = x - 0.5
        for w, b in self.layers:
            x = np.maximum(_conv(x, w, b, 2, 2), 0.0)
        x = x.reshape(-1)
        x = np.maximum(self.fc[0][0] @ x + self.fc[0][1], 0.0)
        y = self.fc[1][0] @ x + self.fc[1][1]
        return (1.0 / (1.0 + np.exp(-y))).astype(np.float32)
