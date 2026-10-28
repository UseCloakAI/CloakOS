"""Runs training jobs in the background and keeps track of trained policies.

One job at a time, as a separate process (`python -m cloakos.rl.train`) so training never
slows the brain. Policies live in <data>/policies/<task>-<stamp>/.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .envs import SENSORS, TASKS

log = logging.getLogger(__name__)


def available() -> str | None:
    """None if training can run, else what's missing."""
    missing = [m for m in ("torch", "stable_baselines3", "gymnasium") if importlib.util.find_spec(m) is None]
    return f"missing {', '.join(missing)} (install the 'rl' extra)" if missing else None


class Trainer:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.root = Path(data_dir) / "policies"
        self.proc: subprocess.Popen | None = None
        self.job: str | None = None

    def running(self) -> str | None:
        if self.proc is not None and self.proc.poll() is not None:
            self.proc, self.job = None, None
        return self.job

    def start(self, task: str, sensors: str = "robot", steps: int = 300_000, envs: int | None = None,
              resume: str | None = None, teacher: str | None = None, real: list[str] | None = None) -> str:
        """resume: a policy of the same skill to keep training (stacking). desk-pixels also takes a
        desk teacher (default: the newest) and real-desk recordings to learn from."""
        if self.running():
            raise RuntimeError("a training run is already going")
        why = available()
        if why:
            raise RuntimeError(why)
        if task not in TASKS:
            raise ValueError(f"unknown task {task}")
        if resume and not (self.root / resume / "meta.json").exists():
            raise ValueError(f"no policy {resume} to build on")
        if task == "desk-pixels":
            teacher = teacher or next((p["name"] for p in self.policies() if p["task"] == "desk" and p["ready"]), None)
            if not teacher:
                raise ValueError("train a desk teacher first (skill: Desk), then the pixel student learns from it")
            name = f"desk-pixels-{time.strftime('%Y%m%d-%H%M%S')}"
            out = self.root / name
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "cloakos.rl.student", "--teacher", str(self.root / teacher),
                   "--steps", str(int(steps)), "--out", str(out)]
            if envs:
                cmd += ["--envs", str(int(envs))]
            if resume:
                cmd += ["--init", str(self.root / resume)]
            if real:
                eps = Path(self.root.parent) / "episodes"
                cmd += ["--real", *[str(eps / Path(r).name) for r in real]]
            return self._launch(name, out, cmd)
        name = f"{task}-{sensors if task != 'balance' else 'robot'}-{time.strftime('%Y%m%d-%H%M%S')}"
        out = self.root / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "cloakos.rl.train", "--task", task, "--sensors",
               sensors if sensors in SENSORS else "robot", "--steps", str(int(steps)), "--out", str(out)]
        if envs:
            cmd += ["--envs", str(int(envs))]
        if resume:
            cmd += ["--resume", str(self.root / resume / "model.zip")]
        return self._launch(name, out, cmd)

    def resume_run(self, name: str) -> str:
        """Pick a stopped or crashed run back up in its own folder, from its last checkpoint."""
        if self.running():
            raise RuntimeError("a training run is already going")
        out = self.root / name
        try:
            meta = json.loads((out / "meta.json").read_text())
        except (OSError, ValueError):
            raise ValueError(f"no run {name}")
        if meta.get("finished"):
            raise ValueError(f"{name} already finished; use Build on to train it more")
        if meta.get("task") == "desk-pixels":
            if not (out / "student.pt").exists():
                raise ValueError("it stopped before its first checkpoint; start it again")
            cmd = [sys.executable, "-m", "cloakos.rl.student", "--teacher", str(self.root / meta["teacher"]),
                   "--out", str(out), "--steps", str(meta["steps"]), "--continue"]
        else:
            if not (out / "model.zip").exists():
                raise ValueError("it stopped before its first checkpoint; start it again")
            cmd = [sys.executable, "-m", "cloakos.rl.train", "--task", meta["task"], "--sensors", meta["sensors"],
                   "--steps", str(meta["steps"]), "--out", str(out), "--resume", str(out / "model.zip")]
        return self._launch(name, out, cmd, append=True)

    def _launch(self, name: str, out: Path, cmd: list[str], append: bool = False) -> str:
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            cmd = ["caffeinate", "-ims", *cmd]           # keep the Mac awake while it trains (lid open / plugged in)
        log.info("training: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=(out / "train.log").open("a" if append else "w"), stderr=subprocess.STDOUT)
        self.job = name
        return name

    def stop(self) -> None:
        if not self.running():
            return
        (self.root / self.job / "STOP").touch()          # finishes the rollout and saves
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.terminate()

    def progress(self, name: str, last: int = 400) -> list[dict]:
        p = self.root / name / "progress.jsonl"
        try:
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        except OSError:
            return []
        step = max(1, len(rows) // last)
        return rows[::step][-last:]

    def policies(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for d in sorted(self.root.iterdir(), reverse=True):
            try:
                meta = json.loads((d / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            rows = self.progress(d.name, last=1)
            out.append({"name": d.name, **{k: meta.get(k) for k in ("task", "sensors", "steps", "trained_steps", "created", "finished",
                                                                    "teacher", "built_on", "resumed_from")},
                        "ready": (d / "policy.npz").exists(), "training": d.name == self.running(),
                        "can_continue": not meta.get("finished") and d.name != self.running()
                        and ((d / "model.zip").exists() or (d / "student.pt").exists()),
                        "last": rows[-1] if rows else None, "log_tail": self._tail(d / "train.log")})
        return out

    @staticmethod
    def _tail(p: Path, n: int = 6) -> str:
        try:
            return "\n".join(p.read_text().splitlines()[-n:])
        except OSError:
            return ""

    def delete_all(self) -> int:
        """Every policy, plus downloaded Nous models and vision training data. Not while training."""
        if self.running():
            raise RuntimeError("stop the training run first")
        n = 0
        if self.root.exists():
            for d in self.root.iterdir():
                shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink()
                n += 1
        from ..nous import override_dir
        for d in (override_dir(), self.root.parent / "vision"):
            shutil.rmtree(d, ignore_errors=True)
        return n

    def delete(self, name: str) -> bool:
        d = (self.root / name).resolve()
        if d.parent != self.root.resolve() or not d.is_dir() or name == self.running():
            return False
        shutil.rmtree(d)
        return True
