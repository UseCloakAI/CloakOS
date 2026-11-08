"""Render the CloakOS app icon into a macOS .iconset folder (the installer runs iconutil on it).

Cloak's look: paper tile, hard ink border and offset shadow, rust orb, with two square
eyes, the same ones Cloak draws on Vector's screen.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PAPER, INK, ACC, SHADOW = (242, 238, 229), (10, 10, 10), (212, 77, 42), (90, 31, 16)


def draw(size: int) -> Image.Image:
    s = 1024
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad, bw, sh = 100, 28, 44
    d.rectangle([pad + sh, pad + sh, s - pad, s - pad], fill=INK)                             # hard shadow
    d.rectangle([pad, pad, s - pad - sh, s - pad - sh], fill=INK)                             # border
    d.rectangle([pad + bw, pad + bw, s - pad - sh - bw, s - pad - sh - bw], fill=PAPER)        # tile
    cx = cy = (pad + s - pad - sh) / 2
    r = 250
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ACC, outline=INK, width=bw)              # orb
    for side in (-1, 1):                                                                       # eyes
        ex, ew, eh = cx + side * 95, 110, 130
        d.rectangle([ex - ew / 2 + 14, cy - eh / 2 + 14, ex + ew / 2 + 14, cy + eh / 2 + 14], fill=SHADOW)
        d.rectangle([ex - ew / 2, cy - eh / 2, ex + ew / 2, cy + eh / 2], fill=PAPER)
    return img.resize((size, size), Image.LANCZOS)


def main(out: str) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    for base in (16, 32, 128, 256, 512):
        draw(base).save(folder / f"icon_{base}x{base}.png")
        draw(base * 2).save(folder / f"icon_{base}x{base}@2x.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AppIcon.iconset")
