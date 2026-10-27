"""The exam: a trained policy on rooms it has never seen, scored the same way every time.

    python -m cloakos.rl.evaluate data/policies/goto-map-...      (writes exam.json next to it)

Seeds come from a range training never uses, the policy runs deterministically through the
numpy runner (exactly what the robot runs), with the same Wi-Fi latency as training.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .envs import VectorEnv, make_spec
from .policy import NumpyPolicy

EXAM_SEEDS = range(900_001, 900_021)        # never used in training (training seeds are small ints)

PASS = {
    "goto": {"success": (">=", 0.85), "bumps_per_run": ("<=", 1.0), "falls": ("<=", 0.05)},
    "wander": {"cells_per_min": (">=", 60), "bumps_per_min": ("<=", 1.5), "falls": ("<=", 0.05)},
}


def run_exam(folder: Path | None, task: str, sensors: str = "map", episodes: int = 20, policy=None) -> dict:
    pol = policy or NumpyPolicy(Path(folder) / "policy.npz")
    env = VectorEnv(task, sensors, seed=0, worlds_per_reload=1)
    spec = make_spec(task, sensors)
    rows = []
    for seed in list(EXAM_SEEDS)[:episodes]:
        obs, _ = env.reset(seed=seed)
        bumps = steps = 0
        info: dict = {}
        while True:
            a, _ = pol.predict(obs)
            obs, _, done, trunc, info = env.step(a)
            steps += 1
            bumps += bool(info.get("bump"))
            if done or trunc:
                break
        minutes = steps / spec.control_hz / 60
        rows.append({"seed": seed, "success": bool(info.get("success")), "fell": bool(info.get("fell")),
                     "bumps": bumps, "seconds": round(steps / spec.control_hz, 1),
                     "cells": info.get("cells"), "minutes": minutes})
    n = len(rows)
    out = {"task": task, "sensors": sensors, "episodes": n, "falls": round(sum(r["fell"] for r in rows) / n, 3)}
    if task == "goto":
        ok = [r for r in rows if r["success"]]
        out.update(success=round(len(ok) / n, 3), bumps_per_run=round(sum(r["bumps"] for r in rows) / n, 2),
                   seconds_to_goal=round(float(np.mean([r["seconds"] for r in ok])), 1) if ok else None)
    elif task == "wander":
        mins = sum(r["minutes"] for r in rows) or 1
        out.update(cells_per_min=round(sum(r["cells"] or 0 for r in rows) / mins, 1),
                   bumps_per_min=round(sum(r["bumps"] for r in rows) / mins, 2))
    marks = PASS.get(task, {})
    out["checks"] = {k: {"value": out.get(k), "want": f"{op} {v}",
                         "ok": out.get(k) is not None and (out[k] >= v if op == ">=" else out[k] <= v)}
                     for k, (op, v) in marks.items()}
    out["passed"] = all(c["ok"] for c in out["checks"].values())
    out["runs"] = rows
    out["when"] = time.time()
    return out


class RandomPolicy:
    def __init__(self, dim: int, seed: int = 0) -> None:
        self.rng, self.dim = np.random.default_rng(seed), dim

    def predict(self, obs, deterministic=True):
        return self.rng.uniform(-1, 1, self.dim).astype(np.float32), None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cloakos.rl.evaluate")
    ap.add_argument("policy", nargs="?", help="policy folder (meta.json + policy.npz)")
    ap.add_argument("--task", help="for --random")
    ap.add_argument("--random", action="store_true", help="score random driving, as a baseline")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args(argv)
    if a.random:
        res = run_exam(None, a.task, "map", a.episodes, policy=RandomPolicy(2))
    else:
        meta = json.loads((Path(a.policy) / "meta.json").read_text())
        res = run_exam(Path(a.policy), meta["task"], meta["sensors"], a.episodes)
        (Path(a.policy) / "exam.json").write_text(json.dumps(res, indent=1))
    summary = {k: v for k, v in res.items() if k not in ("runs",)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
