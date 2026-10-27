"""Procedural eyes for Vector's 184x96 screen: two white rounded rectangles that move like they're
alive. Every parameter eases toward its target with a spring, so expressions blend instead of
snapping; blinks squash, glances overshoot a little, and the eyes breathe when nothing happens.

When Vector starts something new (wandering, going to the mug, undocking...) the words take over the
screen for a moment, big and bold, then the eyes grow back.

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
SS = 3   # supersample for smooth curves
BASE_EYE = 52.0   # eye width in screen pixels (the screen is 184x96)
EYE_GAP = 42.0    # distance of each eye from the middle


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
        self._ov = 0.0                    # 0 = eyes, 1 = words
        self._shown_text = ""
        self._said: str | None = None     # the line already announced (each is shown once)
        self._text_t0: float | None = None
        self._fonts: dict[int, object] = {}
        self._layouts: dict[str, tuple[list[str], int]] = {}
        self._grads: dict[tuple, Image.Image] = {}

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
        self._ov = self._announce(now)
        return self.render(s, blink=blink, breathe=breathe, overlay=self._ov, text=self._shown_text)

    # a new activity's name takes over the screen for a moment, then the eyes come back
    TEXT_IN, TEXT_HOLD, TEXT_OUT = 0.18, 1.5, 0.45

    def _text(self, now: float) -> str | None:
        if self._flash and now < self._flash[1]:
            return self._flash[0]
        self._flash = None
        return self.status

    def _announce(self, now: float) -> float:
        """0 = eyes, 1 = words. Each new line is shown once, eased in and out."""
        text = self._text(now)
        if text and text != self._said:
            self._said, self._shown_text = text, text
            # if words are already up, swap them without blanking
            self._text_t0 = now - (self.TEXT_IN if self._ov > 0.5 else 0.0)
        elif not text:
            self._said = None
        if self._text_t0 is None:
            return 0.0
        t = now - self._text_t0
        if t < self.TEXT_IN:
            x = t / self.TEXT_IN
        elif t < self.TEXT_IN + self.TEXT_HOLD:
            x = 1.0
        elif t < self.TEXT_IN + self.TEXT_HOLD + self.TEXT_OUT:
            x = 1.0 - (t - self.TEXT_IN - self.TEXT_HOLD) / self.TEXT_OUT
        else:
            self._text_t0 = None
            return 0.0
        return x * x * (3 - 2 * x)                       # smoothstep

    def render(self, s: EyeShape, blink: float = 0.0, breathe: float = 0.0, overlay: float = 0.0, text: str = "") -> Image.Image:
        S = SS
        img = Image.new("RGB", (W * S, H * S), self.bg)
        eyes = Image.new("L", (W * S, H * S), 0)              # eye mask, lids cut out of it
        dr = ImageDraw.Draw(eyes)
        jx = random.uniform(-s.jitter, s.jitter) if s.jitter > 0.05 else 0.0
        jy = random.uniform(-s.jitter, s.jitter) if s.jitter > 0.05 else 0.0
        # words take the screen: the eyes shrink away, then grow back with a little pop
        shrink = (1.0 - overlay) * (1.0 + 0.08 * math.sin(math.pi * min(1.0, overlay * 2))) if overlay < 1 else 0.0
        open_ = max(0.0, min(1.2, s.open)) * (1 - blink)
        # squash and stretch: the eyes lengthen along fast gaze moves, like a real glance
        vx, vy = self._vel.get("gaze_x", 0.0), self._vel.get("gaze_y", 0.0)
        stretch_x = 1.0 + min(0.14, abs(vx) * 0.035)
        stretch_y = 1.0 - min(0.08, abs(vx) * 0.02) + min(0.1, abs(vy) * 0.03)
        for side in (-1, 1):
            size = BASE_EYE * max(0.4, s.scale) * (1 + breathe) * shrink * (1 + (s.asym * 0.55 if side > 0 else -s.asym * 0.3))
            w = size * (1 + 0.14 * blink) * stretch_x           # blinks squash a little wider
            h = max(3.0, size * 1.22 * open_ * (1 - 0.25 * max(0.0, s.squint)) * stretch_y)
            # the eye nearer the gaze direction is a touch bigger (cheap perspective)
            persp = 1.0 + 0.05 * s.gaze_x * side
            w, h = w * persp, h * persp
            cx = W / 2 + side * EYE_GAP * shrink + s.gaze_x * 13 + jx
            cy = H / 2 + s.gaze_y * 8 + jy
            box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            r = min(w, h) * 0.36
            dr.rounded_rectangle([v * S for v in box], radius=r * S, fill=255)
            # lids: smooth cuts out of the mask
            if s.happy > 0.05:                                   # lower lid pushes up in an arc: ^ ^
                rr = w * 0.78
                cut_y = box[3] - h * 0.5 * min(1.0, s.happy)
                dr.ellipse([(cx - rr) * S, cut_y * S, (cx + rr) * S, (cut_y + h * 1.5) * S], fill=0)
            inner = box[0] if side > 0 else box[2]
            outer = box[2] if side > 0 else box[0]
            top = box[1] - 4
            if s.sad > 0.05:                                     # upper lid drops toward the outside, with a soft curve
                drop = h * 0.6 * min(1.0, s.sad)
                dr.polygon([(outer * S, top * S), (outer * S, (box[1] + drop) * S),
                            ((inner * 0.35 + outer * 0.65) * S, (box[1] + drop * 0.45) * S), (inner * S, top * S)], fill=0)
            if s.angry > 0.05:
                drop = h * 0.55 * min(1.0, s.angry)
                dr.polygon([(inner * S, top * S), (inner * S, (box[1] + drop) * S),
                            ((outer * 0.35 + inner * 0.65) * S, (box[1] + drop * 0.45) * S), (outer * S, top * S)], fill=0)
        eyes = eyes.filter(ImageFilter.GaussianBlur(0.6 * S))   # softens the cut corners; the downscale does the rest
        bright = 1.0 - overlay
        color = tuple(int(c * bright + b * (1 - bright)) for c, b in zip(self.color, self.bg))
        if shrink > 0.02:
            if self.glow:                                        # soft halo, like a lit screen
                halo = eyes.filter(ImageFilter.GaussianBlur(7 * S)).point(lambda v: int(v * 0.4))
                img.paste(Image.new("RGB", img.size, color), (0, 0), halo)
            # gentle shading: full white on top easing to a soft grey-white at the bottom edge
            grad = self._gradient(color)
            img.paste(grad, (0, 0), eyes)
        if overlay > 0.02 and text:
            self._draw_text(img, text, overlay)
        return img.resize((W, H), Image.LANCZOS)

    def _gradient(self, color) -> Image.Image:
        key = tuple(color)
        g = self._grads.get(key)
        if g is None:
            col = Image.linear_gradient("L").resize((W * SS, H * SS)).rotate(0)      # black top -> white bottom
            lo = Image.new("RGB", (W * SS, H * SS), key)
            hi = Image.new("RGB", (W * SS, H * SS), tuple(int(c * 0.86) for c in key))
            g = Image.composite(hi, lo, col.point(lambda v: int(v * 0.9)))
            self._grads[key] = g
        return g

    def _font(self, px: int):
        if px not in self._fonts:
            try:
                self._fonts[px] = ImageFont.load_default(size=px)
            except Exception:
                self._fonts[px] = ImageFont.load_default()
        return self._fonts[px]

    def _layout(self, text: str) -> tuple[list[str], int]:
        """The biggest size the words fit at, on one line or two (whichever is bigger)."""
        if text in self._layouts:
            return self._layouts[text]
        dr = ImageDraw.Draw(Image.new("L", (1, 1)))
        words = text.split()
        options = [[text]] + [[" ".join(words[:i]), " ".join(words[i:])] for i in range(1, len(words))]
        best: tuple[list[str], int] = ([text], 8 * SS)
        for lines in options:
            lo, hi = 8 * SS, 60 * SS
            while lo < hi:                               # binary search the font size
                mid = (lo + hi + 1) // 2
                f = self._font(mid)
                stroke = max(1, mid // 22)
                ws = [dr.textbbox((0, 0), ln, font=f, stroke_width=stroke) for ln in lines]
                tw = max(b[2] - b[0] for b in ws)
                th = sum(b[3] - b[1] for b in ws) + (len(lines) - 1) * mid * 0.15
                if tw <= (W - 12) * SS and th <= (H - 14) * SS:
                    lo = mid
                else:
                    hi = mid - 1
            if lo > best[1]:
                best = (lines, lo)
        self._layouts[text] = best
        return best

    def _draw_text(self, img: Image.Image, text: str, alpha: float) -> None:
        """Words fill the screen: centred, as big as they fit, bold (stroked), easing in a touch large."""
        lines, px = self._layout(text)
        a = max(0.0, min(1.0, alpha))
        px_now = max(8, int(px * (1.0 + 0.12 * (1 - a))))           # settles from slightly bigger
        f = self._font(px_now)
        stroke = max(1, px_now // 22)
        dr = ImageDraw.Draw(img)
        boxes = [dr.textbbox((0, 0), ln, font=f, stroke_width=stroke) for ln in lines]
        gap = px_now * 0.15
        total = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
        y = (H * SS - total) / 2
        col = tuple(int(c * a + b * (1 - a)) for c, b in zip(self.color, self.bg))
        for ln, b in zip(lines, boxes):
            x = (W * SS - (b[2] - b[0])) / 2 - b[0]
            dr.text((x, y - b[1]), ln, font=f, fill=col, stroke_width=stroke, stroke_fill=col)
            y += (b[3] - b[1]) + gap


def preview_strip(eyes: Eyes, shapes: dict[str, EyeShape]) -> Image.Image:
    """Debug helper: render several shapes side by side."""
    imgs = [eyes.render(s) for s in shapes.values()]
    out = Image.new("RGB", (W * len(imgs) + 6 * (len(imgs) - 1), H), (20, 20, 20))
    for i, im in enumerate(imgs):
        out.paste(im, (i * (W + 6), 0))
    return out
