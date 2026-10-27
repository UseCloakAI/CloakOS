"""Realistic assets for the sim, fetched on demand and cached.

Objects: Google Scanned Objects (1030 photo-scanned household items, CC-BY 4.0), via the
MuJoCo conversion at github.com/kevinzakka/mujoco_scanned_objects. Only the visual mesh and
its texture are downloaded (~1-2 MB each); MuJoCo builds the collision hull from the mesh.

Materials: wood floors, carpet, tile, concrete, painted walls and fabric are generated
procedurally (numpy) into the cache, so worlds look real without any download.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import zlib
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RAW = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/main/models"
LISTING = "https://api.github.com/repos/kevinzakka/mujoco_scanned_objects/contents/models"

# Known-good objects, used until (or instead of) the full catalog: name -> label
STARTER = {
    "ACE_Coffee_Mug_Kristen_16_oz_cup": "mug", "Room_Essentials_Mug_White_Yellow": "mug",
    "Cole_Hardware_Mug_Classic_Blue": "mug", "Threshold_Ramekin_White_Porcelain": "bowl",
    "Cole_Hardware_Bowl_Scirocco_YellowBlue": "bowl", "Threshold_Bead_Cereal_Bowl_White": "bowl",
    "Room_Essentials_Bowl_Turquiose": "bowl", "Threshold_Porcelain_Teapot_White": "teapot",
    "Mens_ASV_Billfish_Boat_Shoe_in_Dark_Brown_Leather_zdHVHXueI3w": "shoe", "11pro_SL_TRX_FG": "shoe",
    "Reebok_SH_PRIME_COURT_LOW": "shoe", "Android_Figure_Panda": "toy", "Nintendo_Mario_Action_Figure": "toy",
    "Squirrel": "toy", "Shark": "toy", "Elephant": "toy", "Dog": "toy", "Racoon": "toy", "Dino_3": "dinosaur",
    "Dino_4": "dinosaur", "Dino_5": "dinosaur", "Great_Dinos_Triceratops_Toy": "dinosaur",
    "Weisshai_Great_White_Shark": "toy", "Schleich_Lion_Action_Figure": "toy", "Schleich_Hereford_Bull": "toy",
    "Schleich_African_Black_Rhino": "toy", "Breyer_Horse_Of_The_Year_2015": "toy",
    "Sootheze_Cold_Therapy_Elephant": "plush", "Sonny_School_Bus": "toy car",
    "Fisher_price_Classic_Toys_Buzzy_Bee": "toy", "Granimals_20_Wooden_ABC_Blocks_Wagon": "toy",
    "Animal_Planet_Foam_2Headed_Dragon": "toy", "Vtech_Stack_Sing_Rings_636_Months": "toy",
    "Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure": "toy",
    "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure": "toy",
    "Marvel_Avengers_Titan_Hero_Series_Doctor_Doom": "toy", "Toysmith_Windem_Up_Flippin_Animals_Dog": "toy",
    "Kong_Puppy_Teething_Rubber_Small_Pink": "dog toy", "Crosley_Alarm_Clock_Vintage_Metal": "clock",
    "Threshold_Basket_Natural_Finish_Fabric_Liner_Small": "basket", "Spritz_Easter_Basket_Plastic_Teal": "basket",
    "Big_Dot_Aqua_Pencil_Case": "pencil case", "Wishbone_Pencil_Case": "pencil case",
    "Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black": "speaker", "Nikon_1_AW1_w11275mm_Lens_Silver": "camera",
    "Chef_Style_Round_Cake_Pan_9_inch_pan": "pan", "Rubbermaid_Large_Drainer": "dish rack",
    "Crayola_Model_Magic_Modeling_Material_White_3_oz": "box", "Pepsi_Caffeine_Free_Diet_12_CT": "box",
    "Sapota_Threshold_4_Ceramic_Round_Planter_Red": "plant pot", "Black_Decker_Stainless_Steel_Toaster_4_Slice": "toaster",
}

LABELS = [  # keyword in the object's name -> what you'd call it
    ("mug", "mug"), ("cup", "cup"), ("bowl", "bowl"), ("plate", "plate"), ("teapot", "teapot"), ("pan", "pan"),
    ("shoe", "shoe"), ("sneaker", "shoe"), ("boot", "shoe"), ("slipper", "shoe"), ("sandal", "shoe"),
    ("dino", "dinosaur"), ("figure", "toy"), ("toy", "toy"), ("lego", "toy"), ("plush", "plush"),
    ("ball", "ball"), ("basket", "basket"), ("clock", "clock"), ("speaker", "speaker"), ("headphone", "headphones"),
    ("bottle", "bottle"), ("can", "can"), ("box", "box"), ("book", "book"), ("game", "box"), ("planter", "plant pot"),
    ("pot", "pot"), ("lamp", "lamp"), ("camera", "camera"), ("lens", "camera"), ("toaster", "toaster"),
    ("pencil", "pencil case"), ("bag", "bag"), ("hat", "hat"), ("glasses", "glasses"),
]


def label_for(name: str) -> str:
    if name in STARTER:
        return STARTER[name]
    n = name.lower()
    for key, label in LABELS:
        if key in n:
            return label
    return "object"


def pretty(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


class Assets:
    """On-demand cache of scanned objects + generated textures."""

    def __init__(self, root: str | os.PathLike = "data/sim_assets", online: bool = True) -> None:
        self.root = Path(root)
        self.online = online
        self._lock = threading.Lock()
        self._catalog: list[str] | None = None
        self._failed: set[str] = set()

    # ── objects ────────────────────────────────────────────────

    def catalog(self) -> list[str]:
        """Every object name we can use: the full list when GitHub answers, else the starter set."""
        if self._catalog is not None:
            return self._catalog
        cache = self.root / "catalog.json"
        names: list[str] = []
        try:
            if cache.exists() and time.time() - cache.stat().st_mtime < 30 * 86400:
                names = json.loads(cache.read_text())
        except (OSError, ValueError):
            names = []
        if not names and self.online:
            try:
                import httpx

                r = httpx.get(LISTING, timeout=10, headers={"Accept": "application/vnd.github+json"})
                r.raise_for_status()
                names = sorted(e["name"] for e in r.json() if e.get("type") == "dir")
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(names))
            except Exception as e:
                log.info("scanned-object catalog unavailable (%s); using the starter set", e)
        self._catalog = sorted(set(names) | set(STARTER))
        return self._catalog

    def cached(self) -> list[str]:
        d = self.root / "objects"
        return sorted(p.name for p in d.iterdir() if (p / "model.obj").exists() and (p / "texture.png").exists()) if d.exists() else []

    def object(self, name: str) -> Path | None:
        """Folder with model.obj + texture.png, downloading it if needed. None if unavailable."""
        d = self.root / "objects" / name
        if (d / "model.obj").exists() and (d / "texture.png").exists():
            return d
        if not self.online or name in self._failed:
            return None
        with self._lock:
            try:
                import httpx

                d.mkdir(parents=True, exist_ok=True)
                for f in ("model.obj", "texture.png"):
                    r = httpx.get(f"{RAW}/{name}/{f}", timeout=30, follow_redirects=True)
                    r.raise_for_status()
                    tmp = d / (f + ".part")
                    tmp.write_bytes(r.content)
                    tmp.replace(d / f)
                return d
            except Exception as e:
                log.info("couldn't fetch scanned object %s: %s", name, e)
                self._failed.add(name)
                return None

    def pick(self, rng: np.random.Generator, n: int, prefer_cached: float = 0.6) -> list[str]:
        """n distinct object names: mostly ones already on disk (fast), some new ones (variety)."""
        have = self.cached()
        pool = self.catalog()
        out: list[str] = []
        for _ in range(n * 3):
            if len(out) >= n:
                break
            src = have if have and rng.random() < prefer_cached else pool
            name = src[int(rng.integers(len(src)))]
            if name not in out and name not in self._failed:
                out.append(name)
        return out

    # ── materials ──────────────────────────────────────────────

    def texture(self, kind: str, variant: int) -> Path:
        """A generated material texture (PNG), made once and cached."""
        p = self.root / "textures" / f"{kind}_{variant}.png"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            img = MATERIALS[kind](np.random.default_rng(zlib.crc32(f"{kind}:{variant}".encode())))
            from PIL import Image

            Image.fromarray(img).save(p)
        return p


# ── procedural materials (512x512 RGB, tileable enough at room scale) ──────────

def _noise(rng, size=512, octaves=5, base=4):
    """Smooth value noise in [0, 1]."""
    from PIL import Image

    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.random((n + 1, n + 1)).astype(np.float32)
        g[-1], g[:, -1] = g[0], g[:, 0]                        # wrap so it tiles
        layer = np.asarray(Image.fromarray(g).resize((size, size), Image.BICUBIC))
        out += amp * layer
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / max(1e-6, out.max() - out.min())


def _tint(gray, color, spread=0.35):
    c = np.array(color, np.float32)[None, None, :]
    return np.clip(c * (1 - spread / 2 + spread * gray[..., None]), 0, 255).astype(np.uint8)


def _wood(rng):
    s = 512
    base = rng.choice([(150, 105, 70), (190, 150, 105), (110, 75, 50), (205, 175, 135), (90, 60, 40)])
    plank_h = s // int(rng.integers(4, 7))
    y = np.arange(s)[:, None]
    x = np.arange(s)[None, :]
    rows = y // plank_h
    offs = (rows * 137) % s
    grain = np.sin((x + offs) * 0.02 + 6 * _noise(rng, s, 4, 2)[y % s, x] + (y % plank_h) * 0.35) * 0.5 + 0.5
    shade = 0.85 + 0.3 * (np.random.default_rng(int(rng.integers(1e9))).random(rows.max() + 2))[rows]
    g = (0.55 * grain + 0.45 * _noise(rng, s, 6, 8)) * shade
    img = _tint(g, base, 0.5)
    seam = (y % plank_h) < 2
    joint = ((x + offs * 3) % (s // 2)) < 2
    img[np.broadcast_to(seam | (joint & ~seam), img.shape[:2])] = (np.array(base) * 0.45).astype(np.uint8)
    return img


def _carpet(rng):
    base = rng.choice([(120, 110, 100), (80, 90, 110), (150, 140, 120), (100, 70, 70), (70, 85, 75), (170, 165, 155)])
    fine = rng.random((512, 512)).astype(np.float32)
    g = 0.6 * fine + 0.4 * _noise(rng, 512, 5, 16)
    return _tint(g, base, 0.35)


def _tile(rng):
    base = rng.choice([(225, 222, 215), (200, 200, 205), (180, 170, 155), (235, 230, 220), (150, 160, 165)])
    n = int(rng.integers(3, 6))
    s = 512
    cell = s // n
    y, x = np.mgrid[0:s, 0:s]
    g = 0.85 + 0.15 * _noise(rng, s, 5, 8)
    img = _tint(g, base, 0.25)
    grout = ((x % cell) < 4) | ((y % cell) < 4)
    img[grout] = (np.array(base) * 0.62).astype(np.uint8)
    return img


def _concrete(rng):
    base = rng.choice([(150, 148, 142), (120, 118, 115), (175, 170, 160)])
    g = 0.5 * _noise(rng, 512, 6, 4) + 0.5 * rng.random((512, 512)).astype(np.float32) * 0.6
    return _tint(g, base, 0.3)


def _paint(rng):
    base = rng.choice([(235, 232, 225), (210, 220, 230), (225, 215, 195), (190, 205, 190), (240, 235, 240),
                       (180, 170, 160), (205, 190, 200), (150, 170, 185)])
    g = 0.7 + 0.3 * _noise(rng, 512, 6, 8)
    return _tint(g, base, 0.08)


def _fabric(rng):
    base = rng.choice([(90, 100, 120), (130, 110, 90), (70, 70, 75), (150, 60, 55), (95, 120, 100), (180, 175, 165)])
    y, x = np.mgrid[0:512, 0:512]
    weave = ((np.sin(x * 0.9) * np.sin(y * 0.9)) * 0.5 + 0.5).astype(np.float32)
    g = 0.5 * weave + 0.5 * _noise(rng, 512, 5, 8)
    return _tint(g, base, 0.3)


MATERIALS = {"wood": _wood, "carpet": _carpet, "tile": _tile, "concrete": _concrete, "paint": _paint, "fabric": _fabric}
FLOORS = ("wood", "wood", "carpet", "tile", "concrete")
