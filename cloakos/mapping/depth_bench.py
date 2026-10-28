"""Score depth models against the simulator's exact depth, on the same camera frames Vector would see.

    python -m cloakos.mapping.depth_bench make  --frames 150 --out data/vision/depth_bench.npz   (no torch)
    python -m cloakos.mapping.depth_bench score data/vision/depth_bench.npz                       (no mujoco)

Two steps because torch and the software renderer can't share a process on some machines.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time

import numpy as np

NEAR, FAR = 0.08, 1.5          # what matters for driving


def make(frames: int, out: str, seed: int) -> None:
    from PIL import Image

    from ..sim.engine import Sim

    rng = np.random.default_rng(seed)
    sim = Sim({"sim": {"assets": False, "camera_fps": 60, "camera_latency_s": 0.0}})
    sim.touch(86400)
    sim.want_depth = True
    rst: dict = {}
    rgb, depth, pos, R, fz = [], [], [], [], []
    while len(rgb) < frames:
        world = str(rng.choice(["room", "room", "table", "person"]))
        sc = sim.load(world, int(rng.integers(1, 1_000_000)))
        hx, hy = sc.room
        for _ in range(15):
            for _try in range(50):
                x, y = rng.uniform(-hx + 0.1, hx - 0.1), rng.uniform(-hy + 0.1, hy - 0.1)
                if world == "table" or all(math.hypot(x - ox, y - oy) > r + 0.12 for ox, oy, r in sc.obstacles):
                    break
            sim.place_robot(float(x), float(y), float(rng.uniform(-math.pi, math.pi)))
            sim.set_head(math.radians(float(rng.uniform(0, 25))))
            sim.step(0.3)
            rst["next"], sim._next_depth = 0.0, 0.0
            sim._render_once(rst)
            sim._publish_frame(time.monotonic() + 1)
            f = sim.depth_frame
            if f is None or sim.frame is None:
                continue
            rgb.append(np.asarray(Image.open(io.BytesIO(sim.frame)).convert("RGB")))
            d = f["depth"]
            depth.append(np.asarray(Image.fromarray(d).resize((640, 360), Image.NEAREST), np.float32))
            pos.append(f["pos"]); R.append(f["R"]); fz.append(f["floor_z"])
        print(f"{len(rgb)}/{frames}", flush=True)
    np.savez_compressed(out, rgb=np.stack(rgb[:frames]), depth=np.stack(depth[:frames]),
                        pos=np.stack(pos[:frames]), R=np.stack(R[:frames]), floor_z=np.array(fz[:frames]))


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    m = np.isfinite(true) & (true > NEAR) & (true < FAR) & np.isfinite(pred) & (pred > 0)
    if m.sum() < 100:
        return {}
    p, t = pred[m], true[m]
    ratio = np.maximum(p / t, t / p)
    return {"abs_rel": float(np.mean(np.abs(p - t) / t)), "within_25pct": float(np.mean(ratio < 1.25)),
            "too_close": float(np.mean(p < 0.7 * t)),            # phantom obstacles
            "too_far": float(np.mean(p > 1.4 * t)),              # missed obstacles
            "coverage": float(m.sum() / max(1, (np.isfinite(true) & (true > NEAR) & (true < FAR)).sum()))}


def score(path: str, models: list[str], limit: int | None = None) -> dict:
    from PIL import Image

    from .vision import CameraPose, DepthModel

    d = np.load(path)
    results = {}
    for name in models:
        dm = DepthModel(name)
        rows, t0 = [], time.time()
        n = min(len(d["rgb"]), limit or len(d["rgb"]))
        for i in range(n):
            img = Image.fromarray(d["rgb"][i])
            cam = CameraPose(d["pos"][i], d["R"][i])
            try:
                z = dm.metric(img, cam, None, float(d["floor_z"][i]))
            except Exception as e:
                rows = [{"error": f"{type(e).__name__}: {e}"}]
                break
            if z is not None:
                from .mapper import clean_depth
                rows.append(_metrics(clean_depth(z), d["depth"][i]))      # what the map actually keeps
        ok = [r for r in rows if r and "error" not in r]
        res = {k: round(float(np.mean([r[k] for r in ok])), 3) for k in ok[0]} if ok else (rows[0] if rows else {})
        res.update(frames_used=len(ok), frames=n, s_per_frame=round((time.time() - t0) / max(1, n), 2))
        results[name] = res
        print(name, json.dumps(res), flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cloakos.mapping.depth_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--frames", type=int, default=150); m.add_argument("--out", default="data/vision/depth_bench.npz"); m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score"); s.add_argument("path"); s.add_argument("--models", nargs="+"); s.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(a.frames, a.out, a.seed)
    else:
        from .vision import DepthModel
        score(a.path, a.models or [DepthModel.NAME], a.limit)


if __name__ == "__main__":
    main()
