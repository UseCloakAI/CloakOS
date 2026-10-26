"""Procedural eyes for Vector's 184x96 screen, in Cloak's look: square eyes, hard offset
shadow, rust accent. Shape comes from the drives; gaze follows the face it sees.

Pure function of (drives, behavior, face, mood, time) -> PIL image, so it's testable and
a learned model could drive the same parameters later.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from PIL import Image, ImageDraw

W, H = 184, 96
SS = 2   # supersample for clean slanted edges


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass
class EyeShape:
    open: float = 1.0        # 0 closed .. 1 open
    scale: float = 1.0
    gaze_x: float = 0.0      # -1 .. 1
    gaze_y: float = 0.0
    happy: float = 0.0       # bottom cut: ^ ^
    sad: float = 0.0         # lids droop toward the outside
    angry: float = 0.0       # lids slant down toward the nose
    jitter: float = 0.0      # px of shake (fear)
    asym: float = 0.0        # one eye bigger (curious)


class Eyes:
    def __init__(self, cfg: dict) -> None:
        c = (cfg.get("expression", {}) or {})
        self.color = _hex(c.get("eye_color", "#D44D2A"))
        self.shadow = _hex(c.get("eye_shadow", "#5A1F10"))
        self.bg = _hex(c.get("screen_bg", "#000000"))
        self._next_blink = time.time() + 3
        self._blink_until = 0.0
        self._wander = (0.0, 0.0)
        self._next_wander = 0.0
        self.mood: str | None = None      # set by emotes for a moment ("happy", "scared", ...)
        self.mood_until = 0.0

    def set_mood(self, mood: str, seconds: float) -> None:
        self.mood, self.mood_until = mood, time.time() + seconds

    def shape(self, d: dict[str, float], behavior: str, face: tuple[float, float] | None, now: float | None = None) -> EyeShape:
        now = now or time.time()
        s = EyeShape()
        s.open = 0.45 + 0.55 * min(1.0, d["energy"] / 0.6)
        s.scale = 0.9 + 0.25 * d["curiosity"] - 0.15 * d["fear"]
        s.happy = max(0.0, d["attachment"] - 0.45) * 1.6 * (1 - d["fear"])
        s.sad = max(0.0, d["boredom"] - 0.4) * 1.4 * (1 - d["attachment"])
        s.jitter = 2.5 * max(0.0, d["fear"] - 0.4)
        s.asym = max(0.0, d["curiosity"] - 0.6) * 0.5

        if behavior == "nap":
            s.open, s.happy, s.sad = 0.06, 0.0, 0.0
        elif behavior == "sulk":
            s.sad, s.gaze_y = max(s.sad, 0.7), 0.6
        elif behavior == "startle":
            s.open, s.scale, s.jitter = 1.0, 1.15, max(s.jitter, 2.0)

        mood = self.mood if now < self.mood_until else None
        if mood == "happy" or mood == "greet":
            s.happy = max(s.happy, 0.8)
        elif mood == "sad":
            s.sad = max(s.sad, 0.8)
        elif mood in ("scared", "startle"):
            s.open, s.scale, s.jitter = 1.0, 1.2, 2.5
        elif mood == "curious":
            s.asym, s.scale = 0.35, s.scale * 1.08
        elif mood == "sleepy":
            s.open = min(s.open, 0.3)
        elif mood == "angry":
            s.angry = 0.7

        if face is not None:
            s.gaze_x = (face[0] - 0.5) * 2
            s.gaze_y = (face[1] - 0.5) * 1.5
        elif behavior != "sulk":
            if now > self._next_wander:
                self._wander = (random.uniform(-0.7, 0.7), random.uniform(-0.4, 0.3))
                self._next_wander = now + random.uniform(1.5, 4)
            s.gaze_x, s.gaze_y = self._wander

        if behavior != "nap":
            if now > self._next_blink:
                self._blink_until = now + 0.12
                self._next_blink = now + random.uniform(2.5, 6)
            if now < self._blink_until:
                s.open = 0.05
        return s

    def render(self, s: EyeShape) -> Image.Image:
        img = Image.new("RGB", (W * SS, H * SS), self.bg)
        dr = ImageDraw.Draw(img)
        jx = random.uniform(-s.jitter, s.jitter) if s.jitter else 0.0
        jy = random.uniform(-s.jitter, s.jitter) if s.jitter else 0.0
        for side in (-1, 1):
            size = 40 * s.scale * (1 + (s.asym if side > 0 else -s.asym * 0.5))
            w, h = size, max(3.0, size * 1.12 * s.open)
            cx = W / 2 + side * 38 + s.gaze_x * 16 + jx
            cy = H / 2 + s.gaze_y * 10 + jy
            box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            # hard offset shadow first (Cloak's "the shadow is the floor"), then the eye
            sh = 4
            self._rect(dr, [box[0] + sh, box[1] + sh, box[2] + sh, box[3] + sh], self.shadow)
            self._rect(dr, box, self.color)
            # lids: cut shapes in background colour
            if s.happy > 0.05:
                r = w * 0.75
                cut_y = box[3] - h * 0.42 * min(1.0, s.happy)
                self._ellipse(dr, [cx - r, cut_y, cx + r, cut_y + h * 1.3], self.bg)
            if s.sad > 0.05:
                drop = h * 0.55 * min(1.0, s.sad)
                inner = box[0] if side > 0 else box[2]
                outer = box[2] if side > 0 else box[0]
                self._poly(dr, [(outer, box[1] - 2), (outer, box[1] + drop), (inner, box[1] - 2)], self.bg)
            if s.angry > 0.05:
                drop = h * 0.5 * s.angry
                inner = box[0] if side > 0 else box[2]
                outer = box[2] if side > 0 else box[0]
                self._poly(dr, [(inner, box[1] - 2), (inner, box[1] + drop), (outer, box[1] - 2)], self.bg)
        return img.resize((W, H), Image.LANCZOS)

    @staticmethod
    def _rect(dr: ImageDraw.ImageDraw, b: list[float], fill) -> None:
        dr.rectangle([v * SS for v in b], fill=fill)

    @staticmethod
    def _ellipse(dr: ImageDraw.ImageDraw, b: list[float], fill) -> None:
        dr.ellipse([v * SS for v in b], fill=fill)

    @staticmethod
    def _poly(dr: ImageDraw.ImageDraw, pts: list[tuple[float, float]], fill) -> None:
        dr.polygon([(x * SS, y * SS) for x, y in pts], fill=fill)


def preview_strip(eyes: Eyes, shapes: dict[str, EyeShape]) -> Image.Image:
    """Debug helper: render several shapes side by side."""
    imgs = [eyes.render(s) for s in shapes.values()]
    out = Image.new("RGB", (W * len(imgs) + 6 * (len(imgs) - 1), H), (20, 20, 20))
    for i, im in enumerate(imgs):
        out.paste(im, (i * (W + 6), 0))
    return out

