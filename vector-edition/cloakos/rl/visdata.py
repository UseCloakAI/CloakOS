"""Make training pictures for the vision net: what Vector's camera sees in random rooms, paired with
the true 11 range readings (obstacles and table edges). Runs without torch.

    python -m cloakos.rl.visdata --frames 20000 --out data/vision/train.npz --seed 1
"""
from __future__ import annotations

import argparse
import io
import math
import time
from pathlib import Path

import numpy as np

from .envs import RAY_FOV, RAY_MAX, edge_rays

IMG_W, IMG_H = 96, 54
N_RAYS = 11


def main(argv: list[str] | None = None) -> None:
    import faulthandler
    faulthandler.enable()
    ap = argparse.ArgumentParser(prog="cloakos.rl.visdata")
    ap.add_argument("--frames", type=int, default=5000)
    ap.add_argument("--out", default="data/vision/train.npz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-room", type=int, default=40)
    ap.add_argument("--assets", action="store_true", help="use photo-scanned objects (downloads on first use)")
    a = ap.parse_args(argv)

    from PIL import Image
    from ..sim.engine import Sim

    rng = np.random.default_rng(a.seed)
    sim = Sim({"sim": {"assets": a.assets, "camera_fps": 60, "camera_latency_s": 0.0, "render_width": 480, "render_height": 270}})
    sim.touch(86400)
    rst: dict = {}
    imgs, labels, t0 = [], [], time.time()
    while len(imgs) < a.frames:
        world = str(rng.choice(["room", "room", "room", "table", "table", "empty", "person"]))
        sc = sim.load(world, int(rng.integers(1, 1_000_000)))
        hx, hy = sc.room
        for _ in range(a.per_room):
            for _try in range(50):
                x, y = rng.uniform(-hx + 0.1, hx - 0.1), rng.uniform(-hy + 0.1, hy - 0.1)
                if world == "table" or all(math.hypot(x - ox, y - oy) > r + 0.12 for ox, oy, r in sc.obstacles):
                    break
            sim.place_robot(float(x), float(y), float(rng.uniform(-math.pi, math.pi)))
            sim.set_head(math.radians(float(rng.uniform(5, 40))))
            sim.step(0.3)                                            # let it settle
            rst["next"] = 0.0
            sim._render_once(rst)                                    # one camera frame, same effects as the live feed
            sim._publish_frame(time.monotonic() + 1)
            s = sim.snapshot
            if sim.frame is None or s["upright"] < 0.9:
                continue
            truth = np.minimum(sim.rays(N_RAYS, RAY_FOV, RAY_MAX), edge_rays(s, sc.room, N_RAYS) if world == "table" else RAY_MAX)
            im = Image.open(io.BytesIO(sim.frame)).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
            imgs.append(np.asarray(im, np.uint8))
            labels.append(np.clip(truth, 0, RAY_MAX).astype(np.float16))
        print(f"{len(imgs)}/{a.frames} frames, {len(imgs) / (time.time() - t0):.1f}/s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, x=np.stack(imgs[:a.frames]), y=np.stack(labels[:a.frames]))


if __name__ == "__main__":
    main()
