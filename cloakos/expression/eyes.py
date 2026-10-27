"""Procedural eyes for Vector's 184x96 screen: two white rounded rectangles that move like they're
alive. Every parameter eases toward its target with a spring, so expressions blend instead of
snapping; blinks squash, glances overshoot a little, and the eyes breathe when nothing happens.

When Vector is busy with something (wandering, going to the mug, undocking...) the eyes fade back
and a short line of text says what it's doing.

shape() picks the target from drives/behavior/face/mood; step() moves the live shape toward it;
render() draws a frame. All pure PIL, so it's testable.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, fields

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 184, 96
SS = 2   # supersample for smooth curves


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
    squint: float = 0.0      # both lids in a little (focused)


# how fast each part follows its target (1/s): gaze is quick, moods melt in slowly
SPEED = {"open": 22.0, "scale": 9.0, "gaze_x": 13.0, "gaze_y": 13.0, "happy": 7.0, "sad": 6.0,
         "angry": 8.0, "jitter": 10.0, "asym": 6.0, "squint": 8.0}


class Eyes:
    def __init__(self, cfg: dict) -> None:
        c = (cfg.get("expression", {}) or {})
        self.color = _hex(c.get("eye_color", "#FFFFFF"))
        self.bg = _hex(c.get("screen_bg", "#000000"))
        self.glow = bool(c.get("eye_glow", True))
        self.live = EyeShape()                    # what's on screen right now
        self._vel = {f.name: 0.0 for f in fields(EyeShape)}
        self._last = time.time()
        self._next_blink = time.time() + 2.5
        self._blink_t0: float | None = None
        self._double = False
        self._wander = (0.0, 0.0)
        self._next_wander = 0.0
        self.mood: str | None = None      # set by emotes for a moment ("happy", "scared", ...)
        self.frozen = False               # freeze mode: eyes hold still, half-lidded, no blinking
        self.mood_until = 0.0
        # status overlay: what it's doing, in words
        self.status: str | None = None    # set every frame by the brain
        self._flash: tuple[str, float] | None = None
        self._ov = 0.0                    # overlay fade 0..1
        self._shown_text = ""
        try:
            self.font = ImageFont.load_default(size=14 * SS)
        except Exception:                 # very old Pillow / no FreeType: the small bitmap font
            self.font = ImageFont.load_default()

    def set_mood(self, mood: str, seconds: float) -> None:
        self.mood, self.mood_until = mood, time.time() + seconds

    def flash(self, text: str, seconds: float = 2.0) -> None:
        """Show a line for a moment, over whatever status there is ("Got there!", "On it")."""
        self._flash = (text, time.time() + seconds)

    # ── what the eyes should look like ─────────────────────────

    def shape(self, d: dict[str, float], behavior: str, face: tuple[float, float] | None, now: float | None = None) -> EyeShape:
        now = now or time.time()
        s = EyeShape()
        s.open = 0.5 + 0.5 * min(1.0, d["energy"] / 0.6)
        s.scale = 0.92 + 0.2 * d["curiosity"] - 0.12 * d["fear"]
        s.happy = max(0.0, d["attachment"] - 0.45) * 1.6 * (1 - d["fear"])
        s.sad = max(0.0, d["boredom"] - 0.4) * 1.4 * (1 - d["attachment"])
        s.jitter = 2.0 * max(0.0, d["fear"] - 0.4)
        s.asym = max(0.0, d["curiosity"] - 0.6) * 0.5

        if behavior == "nap":
            s.open, s.happy, s.sad = 0.04, 0.0, 0.0
        elif behavior == "sulk":
            s.sad, s.gaze_y = max(s.sad, 0.7), 0.6
        elif behavior == "startle":
            s.open, s.scale, s.jitter = 1.0, 1.15, max(s.jitter, 2.0)
        elif behavior in ("wander", "goto") or behavior.startswith("going to"):
            s.squint = 0.15                       # a little focused while driving

        if self.frozen:
            return EyeShape(open=0.35, scale=0.95)

        mood = self.mood if now < self.mood_until else None
        if mood in ("happy", "greet"):
            s.happy, s.scale = max(s.happy, 0.85), s.scale * 1.05
        elif mood == "sad":
            s.sad, s.gaze_y = max(s.sad, 0.8), 0.4
        elif mood in ("scared", "startle"):
            s.open, s.scale, s.jitter = 1.0, 1.18, 2.2
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
            if now > self._next_wander:          # quick glances, held for a beat
                self._wander = (random.uniform(-0.75, 0.75), random.uniform(-0.45, 0.35))
                if random.random() < 0.35:
                    self._wander = (0.0, 0.0)        # often back to centre
                self._next_wander = now + random.uniform(0.8, 3.5)
            s.gaze_x, s.gaze_y = self._wander
        return s

    def step(self, target: EyeShape, now: float | None = None) -> EyeShape:
        """Spring the live shape toward the target (slightly underdamped: a touch of overshoot)."""
        now = now or time.time()
        dt = min(0.1, max(0.0, now - self._last))
        self._last = now
        n = max(1, int(dt / 0.005))                      # small substeps: smooth and stable at any frame rate
        h = dt / n
        for f in fields(EyeShape):
            k = SPEED[f.name]
            x, v, t = getattr(self.live, f.name), self._vel[f.name], getattr(target, f.name)
            for _ in range(n):
                a = k * k * (t - x) - 1.6 * k * v      # damping ratio 0.8
                v += a * h
                x += v * h
            self._vel[f.name] = v
            setattr(self.live, f.name, x)
        return self.live

    def _blink(self, now: float, sleeping: bool) -> float:
        """0 = open, 1 = shut. Quick close, slower open; sometimes a double blink."""
        if self.frozen or sleeping:
            return 0.0
        if self._blink_t0 is None and now > self._next_blink:
            self._blink_t0 = now
            self._double = random.random() < 0.2
        if self._blink_t0 is None:
            return 0.0
        t = now - self._blink_t0
        close, hold, opening = 0.06, 0.03, 0.11
        if t < close:
            return t / close
        if t < close + hold:
            return 1.0
        if t < close + hold + opening:
            return 1.0 - (t - close - hold) / opening
        if self._double:
            self._double, self._blink_t0 = False, now + 0.08
            return 0.0
        self._blink_t0 = None
        self._next_blink = now + random.uniform(2.0, 5.5)
        return 0.0

    # ── drawing ────────────────────────────────────────────────

    def frame(self, target: EyeShape, now: float | None = None) -> Image.Image:
        """One animation frame: step toward the target, add life (blink, breathing), draw."""
        now = now or time.time()
        dt = min(0.1, max(0.0, now - self._last))
        s = self.step(target, now)
        sleeping = target.open < 0.1
        blink = self._blink(now, sleeping)
        breathe = 0.0 if self.frozen else 0.018 * math.sin(now * 2 * math.pi * 0.22)
        text = self._text(now)
        goal = 1.0 if text else 0.0
        self._ov += (goal - self._ov) * min(1.0, 7.0 * dt)
        if text:
            self._shown_text = text
        return self.render(s, blink=blink, breathe=breathe, overlay=self._ov, text=self._shown_text)

    def _text(self, now: float) -> str | None:
        if self._flash and now < self._flash[1]:
            return self._flash[0]
        self._flash = None
        return self.status

    def render(self, s: EyeShape, blink: float = 0.0, breathe: float = 0.0, overlay: float = 0.0, text: str = "") -> Image.Image:
        img = Image.new("RGB", (W * SS, H * SS), self.bg)
        eyes = Image.new("L", (W * SS, H * SS), 0)            # eye mask, lids cut out of it
        dr = ImageDraw.Draw(eyes)
        jx = random.uniform(-s.jitter, s.jitter) if s.jitter > 0.05 else 0.0
        jy = random.uniform(-s.jitter, s.jitter) if s.jitter > 0.05 else 0.0
        shrink = 1.0 - 0.22 * overlay                         # eyes step back when there's text
        lift = -14 * overlay
        open_ = max(0.0, min(1.2, s.open)) * (1 - blink)
        for side in (-1, 1):
            size = 38 * max(0.4, s.scale) * (1 + breathe) * shrink * (1 + (s.asym if side > 0 else -s.asym * 0.5))
            w = size * (1 + 0.12 * blink)                      # blinks squash a little wider
            h = max(2.5, size * 1.15 * open_ * (1 - 0.25 * max(0.0, s.squint)))
            cx = W / 2 + side * 36 * shrink + s.gaze_x * 15 + jx
            cy = H / 2 + s.gaze_y * 10 + jy + lift
            box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            r = min(w, h) * 0.32
            dr.rounded_rectangle([v * SS for v in box], radius=r * SS, fill=255)
            # lids: cut shapes out of the mask
            if s.happy > 0.05:
                rr = w * 0.8
                cut_y = box[3] - h * 0.45 * min(1.0, s.happy)
                dr.ellipse([(cx - rr) * SS, cut_y * SS, (cx + rr) * SS, (cut_y + h * 1.4) * SS], fill=0)
            inner = box[0] if side > 0 else box[2]
            outer = box[2] if side > 0 else box[0]
            if s.sad > 0.05:
                drop = h * 0.55 * min(1.0, s.sad)
                dr.polygon([(outer * SS, (box[1] - 3) * SS), (outer * SS, (box[1] + drop) * SS), (inner * SS, (box[1] - 3) * SS)], fill=0)
            if s.angry > 0.05:
                drop = h * 0.5 * min(1.0, s.angry)
                dr.polygon([(inner * SS, (box[1] - 3) * SS), (inner * SS, (box[1] + drop) * SS), (outer * SS, (box[1] - 3) * SS)], fill=0)
        bright = 1.0 - 0.6 * overlay
        color = tuple(int(c * bright + b * (1 - bright)) for c, b in zip(self.color, self.bg))
        if self.glow:                                          # soft halo, like a lit screen
            halo = eyes.filter(ImageFilter.GaussianBlur(5 * SS)).point(lambda v: int(v * 0.35))
            img.paste(Image.new("RGB", img.size, color), (0, 0), halo)
        img.paste(Image.new("RGB", img.size, color), (0, 0), eyes)
        if overlay > 0.02 and text:
            self._draw_text(img, text, overlay)
        return img.resize((W, H), Image.LANCZOS)

    def _draw_text(self, img: Image.Image, text: str, alpha: float) -> None:
        dr = ImageDraw.Draw(img)
        t = text if len(text) <= 22 else text[:21] + "…"
        x0, y0, x1, y1 = dr.textbbox((0, 0), t, font=self.font)
        tw = x1 - x0
        x = (W * SS - tw) / 2 - x0
        y = H * SS - (y1 - y0) - 12 * SS + (1 - alpha) * 8 * SS   # slides up as it fades in
        a = max(0.0, min(1.0, alpha))
        col = tuple(int(c * a + b * (1 - a)) for c, b in zip(self.color, self.bg))
        dr.text((x, y), t, font=self.font, fill=col)


def preview_strip(eyes: Eyes, shapes: dict[str, EyeShape]) -> Image.Image:
    """Debug helper: render several shapes side by side."""
    imgs = [eyes.render(s) for s in shapes.values()]
    out = Image.new("RGB", (W * len(imgs) + 6 * (len(imgs) - 1), H), (20, 20, 20))
    for i, im in enumerate(imgs):
        out.paste(im, (i * (W + 6), 0))
    return out
