"""Train a policy with PPO on the sim:  python -m cloakos.rl.train --task goto --steps 300000

Runs headless, several sims in parallel (one per CPU core by default), much faster than real
time. Writes <out>/progress.jsonl as it goes and <out>/model.zip + meta.json at checkpoints,
so the UI can chart it and the brain can load the result.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def export_numpy(model, path: Path) -> None:
    """The deterministic policy as plain arrays (MLP: tanh hidden layers -> linear mean action),
    so the brain can run it with numpy alone, no torch in the robot process."""
    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if hasattr(m, "weight")] + [pol.action_net]
    arrays = {}
    for i, l in enumerate(layers):
        arrays[f"w{i}"] = l.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = l.bias.detach().cpu().numpy()
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def save_model(model, out: Path) -> None:
    """Write-then-swap, so a crash mid-save never leaves a broken checkpoint."""
    model.save(out / "model.tmp.zip")
    (out / "model.tmp.zip").replace(out / "model.zip")
    export_numpy(model, out / "policy.npz")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cloakos.rl.train")
    ap.add_argument("--task", default="goto")
    ap.add_argument("--sensors", default="robot")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--envs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="model.zip to continue from")
    a = ap.parse_args(argv)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from .envs import VectorEnv, make_spec

    torch.set_num_threads(1)                 # the parallel sims want the cores, not the tiny network
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = make_spec(a.task, a.sensors)
    same_run = bool(a.resume) and Path(a.resume).resolve().parent == out.resolve()
    prev_meta = json.loads((out / "meta.json").read_text()) if same_run and (out / "meta.json").exists() else None
    meta = {"task": a.task, "sensors": spec.sensors, "obs_size": spec.obs_size, "control_hz": spec.control_hz,
            "max_speed": spec.max_speed, "n_rays": spec.n_rays, "steps": a.steps, "envs": a.envs,
            "created": time.time(), "algo": "PPO",
            "resumed_from": Path(a.resume).parent.name if a.resume else None}
    if prev_meta:                    # picking a stopped run back up: same folder, same goal
        meta = {**prev_meta, "continued": prev_meta.get("continued", 0) + 1}
        meta.pop("finished", None)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    progress = (out / "progress.jsonl").open("a")

    print("starting the simulator rooms", flush=True)
    vec = make_vec_env(VectorEnv, n_envs=a.envs, seed=a.seed, vec_env_cls=SubprocVecEnv if a.envs > 1 else None,
                       env_kwargs={"task": a.task, "sensors": a.sensors})

    class Report(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.t0 = time.time()
            self.eps: list[dict] = []
            self.last_save = time.time()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep:
                    self.eps.append({"r": ep["r"], "l": ep["l"], "success": bool(info.get("success")),
                                     "fell": bool(info.get("fell")), "cells": info.get("cells"),
                                     "bumps": info.get("bumps", 0)})
            if (out / "STOP").exists():
                return False
            return True

        def _on_rollout_end(self) -> None:
            recent = self.eps[-100:]
            row = {"t": round(time.time() - self.t0, 1), "steps": int(self.num_timesteps),
                   "fps": int(self.num_timesteps / max(1e-6, time.time() - self.t0)),
                   "episodes": len(self.eps)}
            if recent:
                row["reward"] = round(float(np.mean([e["r"] for e in recent])), 3)
                row["length"] = round(float(np.mean([e["l"] for e in recent])), 1)
                row["success"] = round(float(np.mean([e["success"] for e in recent])), 3)
                row["fell"] = round(float(np.mean([e["fell"] for e in recent])), 3)
                row["bumps"] = round(float(np.mean([e["bumps"] for e in recent])), 2)
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            if time.time() - self.last_save > 60:          # a checkpoint a minute: a dead Mac loses at most that
                save_model(self.model, out)
                self.last_save = time.time()

    if a.resume:
        model = PPO.load(a.resume, env=vec, device="cpu")
        print(f"continuing from {model.num_timesteps} steps", flush=True)
    else:
        model = PPO("MlpPolicy", vec, device="cpu", seed=a.seed, verbose=0,
                    n_steps=1024 if a.task != "balance" else 512, batch_size=256, gamma=0.99 if a.task != "balance" else 0.98,
                    learning_rate=3e-4, ent_coef=0.005, policy_kwargs={"net_arch": [128, 128]})
    try:
        start = int(model.num_timesteps)
        todo = max(0, int(meta["steps"]) - start) if prev_meta else a.steps
        model.learn(total_timesteps=todo, callback=Report(), reset_num_timesteps=not a.resume)
    finally:
        save_model(model, out)
        done = int(model.num_timesteps) - start >= todo        # stopped early or killed: left open to Continue
        if done:
            meta["finished"] = time.time()
        (out / "STOP").unlink(missing_ok=True)
        meta["trained_steps"] = int(model.num_timesteps)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        progress.close()
        vec.close()
    if done:
        finish(out)


def finish(out: Path) -> None:
    """After a run: take the exam (own process, no torch), then share the result."""
    import subprocess
    import sys

    if meta_task(out) in ("goto", "wander", "desk"):
        print("taking the exam", flush=True)
        subprocess.run([sys.executable, "-m", "cloakos.rl.evaluate", str(out)], capture_output=True, timeout=1800)
    from .. import share
    if share.enabled():
        try:
            share.upload_run(out, os.environ.get("CLOAK_SHARE_SOURCE", "local"))
        except Exception as e:                         # offline or Supabase down: training still counts
            print(f"couldn't share this run: {e}", flush=True)


def meta_task(out: Path) -> str | None:
    try:
        return json.loads((out / "meta.json").read_text()).get("task")
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    main()
