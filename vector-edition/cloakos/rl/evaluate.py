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
    "desk": {"cells_per_min": (">=", 40), "bumps_per_min": ("<=", 1.5), "falls": ("<=", 0.05)},
    "wander": {"cells_per_min": (">=", 60), "bumps_per_min": ("<=", 1.5), "falls": ("<=", 0.05)},
}


def _drive(job: tuple, pol=None) -> list[dict]:
    """Exam episodes on some seeds (a worker process: numpy + MuJoCo only)."""
    path, task, sensors, seeds = job
    pol = pol or NumpyPolicy(Path(path))
    env = VectorEnv(task, sensors, seed=0, worlds_per_reload=1)
    spec = make_spec(task, sensors)
    rows = []
    for seed in seeds:
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
    return rows


def _pool_rows(jobs: list[tuple]) -> list[list[dict]]:
    import multiprocessing as mp
    import os

    n = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if n == 1:
        return [_drive(j) for j in jobs]
    with mp.get_context("spawn").Pool(n) as pool:
        return pool.map(_drive, jobs)


def run_exam(folder: Path | None, task: str, sensors: str = "map", episodes: int = 20, policy=None,
             npz: Path | None = None) -> dict:
    """The exam: the same 20 unseen worlds every time, split across CPU cores."""
    seeds = list(EXAM_SEEDS)[:episodes]
    path = str(npz or (Path(folder) / "policy.npz")) if (npz or folder) and policy is None else None
    if policy is not None:                                        # an in-process policy object (tests, baselines)
        rows = _drive((None, task, sensors, seeds), pol=policy)
    else:
        k = max(1, min(8, len(seeds)))
        rows = [r for part in _pool_rows([(path, task, sensors, seeds[i::k]) for i in range(k)]) for r in part]
        rows.sort(key=lambda r: r["seed"])
    return summarize(task, sensors, rows)


def summarize(task: str, sensors: str, rows: list[dict]) -> dict:
    n = len(rows)
    out = {"task": task, "sensors": sensors, "episodes": n, "falls": round(sum(r["fell"] for r in rows) / n, 3)}
    if task == "goto":
        ok = [r for r in rows if r["success"]]
        out.update(success=round(len(ok) / n, 3), bumps_per_run=round(sum(r["bumps"] for r in rows) / n, 2),
                   seconds_to_goal=round(float(np.mean([r["seconds"] for r in ok])), 1) if ok else None)
    elif task in ("wander", "desk"):
        mins = sum(r["minutes"] for r in rows) or 1
        out.update(cells_per_min=round(sum(r["cells"] or 0 for r in rows) / mins, 1),
                   bumps_per_min=round(sum(r["bumps"] for r in rows) / mins, 2))
    marks = PASS.get(task, {})
    out["checks"] = {k: {"value": out.get(k), "want": f"{op} {v}",
                         "ok": out.get(k) is not None and (out[k] >= v if op == ">=" else out[k] <= v)}
                     for k, (op, v) in marks.items()}
    out["passed"] = all(c["ok"] for c in out["checks"].values())
    out["score"] = score(task, out)
    out["hard"] = [r["seed"] for r in rows if r["fell"] or r["bumps"] >= 3 or (task == "goto" and not r["success"])]
    out["runs"] = rows
    out["when"] = time.time()
    return out


def score(task: str, ex: dict) -> float:
    """One number to compare two exams of the same skill (passing first, then how well)."""
    base = 1000.0 if ex.get("passed") else 0.0
    if task == "goto":
        return base + 100 * ex.get("success", 0) - 10 * ex.get("bumps_per_run", 0) - 200 * ex.get("falls", 0)
    return base + ex.get("cells_per_min", 0) - 20 * ex.get("bumps_per_min", 0) - 400 * ex.get("falls", 0)


def pick(folder: Path, episodes: int = 20) -> dict:
    """End of a run: examine the last policy and the best one seen in training, keep the better
    as policy.npz, and compare with the run it was built on (the champion it has to beat)."""
    meta = json.loads((folder / "meta.json").read_text())
    task, sensors = meta["task"], meta["sensors"]
    cands = {"last": folder / "policy.npz"}
    if (folder / "best" / "policy.npz").exists():
        cands["best"] = folder / "best" / "policy.npz"
    seeds = list(EXAM_SEEDS)[:episodes]
    k = max(1, min(8 // len(cands), len(seeds)))
    jobs = [(str(p), task, sensors, seeds[i::k]) for p in cands.values() for i in range(k)]
    parts = _pool_rows(jobs)
    exams = {}
    for j, name in enumerate(cands):
        rows = sorted((r for part in parts[j * k:(j + 1) * k] for r in part), key=lambda r: r["seed"])
        exams[name] = summarize(task, sensors, rows)
    win = max(exams, key=lambda n: exams[n]["score"])
    if win == "best":
        import shutil
        shutil.copyfile(cands["best"], folder / "policy.tmp.npz")
        (folder / "policy.tmp.npz").replace(folder / "policy.npz")
    res = {**exams[win], "picked": win,
           "candidates": {n: {k2: e[k2] for k2 in ("score", "passed", "falls") if k2 in e} for n, e in exams.items()}}
    if win == "best":
        try:
            res["picked_steps"] = json.loads((folder / "best" / "best.json").read_text()).get("steps")
        except (OSError, ValueError):
            pass
    parent = meta.get("built_on") or meta.get("resumed_from")
    if parent and parent != folder.name:
        try:
            pex = json.loads((folder.parent / parent / "exam.json").read_text())
            pscore = pex.get("score", score(task, pex))
            res["parent"] = {"name": parent, "score": round(pscore, 1), "passed": pex.get("passed")}
            res["beats_parent"] = res["score"] > pscore
        except (OSError, ValueError):
            pass
    (folder / "exam.json").write_text(json.dumps(res, indent=1))
    return res


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
    ap.add_argument("--pick", action="store_true", help="exam last and best, keep the better (end of a training run)")
    a = ap.parse_args(argv)
    if a.random:
        res = run_exam(None, a.task, "map", a.episodes, policy=RandomPolicy(2))
    elif a.pick:
        res = pick(Path(a.policy), a.episodes)
    else:
        meta = json.loads((Path(a.policy) / "meta.json").read_text())
        res = run_exam(Path(a.policy), meta["task"], meta["sensors"], a.episodes)
        (Path(a.policy) / "exam.json").write_text(json.dumps(res, indent=1))
    summary = {k: v for k, v in res.items() if k not in ("runs",)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
