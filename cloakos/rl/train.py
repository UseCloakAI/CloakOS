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
    meta = {"task": a.task, "sensors": spec.sensors, "obs_size": spec.obs_size, "control_hz": spec.control_hz,
            "max_speed": spec.max_speed, "n_rays": spec.n_rays, "steps": a.steps, "envs": a.envs,
            "created": time.time(), "algo": "PPO"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    progress = (out / "progress.jsonl").open("a")

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
            if time.time() - self.last_save > 60:
                self.model.save(out / "model.zip")
                export_numpy(self.model, out / "policy.npz")
                self.last_save = time.time()

    if a.resume:
        model = PPO.load(a.resume, env=vec, device="cpu")
    else:
        model = PPO("MlpPolicy", vec, device="cpu", seed=a.seed, verbose=0,
                    n_steps=1024 if a.task != "balance" else 512, batch_size=256, gamma=0.99 if a.task != "balance" else 0.98,
                    learning_rate=3e-4, ent_coef=0.005, policy_kwargs={"net_arch": [128, 128]})
    try:
        model.learn(total_timesteps=a.steps, callback=Report(), reset_num_timesteps=not a.resume)
    finally:
        model.save(out / "model.zip")
        export_numpy(model, out / "policy.npz")
        meta["finished"] = time.time()
        meta["trained_steps"] = int(model.num_timesteps)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        progress.close()
        vec.close()
    finish(out)


def finish(out: Path) -> None:
    """After a run: take the exam (own process, no torch), then share the result."""
    import subprocess
    import sys

    if meta_task(out) in ("goto", "wander"):
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
