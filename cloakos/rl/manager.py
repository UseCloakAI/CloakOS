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
import threading
import time
from pathlib import Path
from typing import Any

from .envs import SENSORS, TASKS, make_spec

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
        self.auto_file = Path(data_dir) / "keep_training.json"
        try:
            self.auto: dict[str, Any] = json.loads(self.auto_file.read_text())
        except (OSError, ValueError):
            self.auto = {"on": False}
        self.started = 0.0
        self.stopping = False

    def running(self) -> str | None:
        if self.proc is not None and self.proc.poll() is not None:
            self.proc, self.job, self.stopping = None, None, False
        return self.job

    def start(self, task: str, sensors: str = "robot", steps: int = 300_000, envs: int | None = None,
              resume: str | None = None, teacher: str | None = None, real: list[str] | None = None,
              hours: float | None = None) -> str:
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
            if hours:
                cmd += ["--hours", str(float(hours))]
            return self._launch(name, out, cmd)
        name = f"{task}-{make_spec(task, sensors).sensors}-{time.strftime('%Y%m%d-%H%M%S')}"
        out = self.root / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "cloakos.rl.train", "--task", task, "--sensors",
               sensors if sensors in SENSORS else "robot", "--steps", str(int(steps)), "--out", str(out)]
        if envs:
            cmd += ["--envs", str(int(envs))]
        if resume:
            cmd += ["--resume", str(self.root / resume / "model.zip")]
        if hours:
            cmd += ["--hours", str(float(hours))]
        return self._launch(name, out, cmd)

    # ── keep training: sessions back to back until you stop it ──

    SESSION_H = 2.0                    # each session ends with a test, so the best driver is always kept

    def set_auto(self, on: bool) -> None:
        self.auto = {"on": on, "since": time.time(), "sessions": 0, "trained_s": 0.0} if on else {**self.auto, "on": False}
        self._save_auto()

    def _save_auto(self) -> None:
        try:
            self.auto_file.parent.mkdir(parents=True, exist_ok=True)
            self.auto_file.write_text(json.dumps(self.auto))
        except OSError:
            pass

    def auto_tick(self, recordings: list[dict]) -> str | None:
        """Called every so often: if Keep training is on and nothing's running, start the next session
        (whatever step is next). A session that dies within minutes backs off for 10 minutes."""
        if not self.auto.get("on") or self.running():
            return None
        now, last = time.time(), self.auto.get("last")
        if now < self.auto.get("wait_until", 0):
            return None
        if last and last.get("name"):
            ran = now - last["start"]
            self.auto["trained_s"] = self.auto.get("trained_s", 0.0) + ran
            last["name"] = None                            # counted
            if ran < 300:                                  # died within minutes: don't hammer it
                self.auto["wait_until"] = now + 600
                self._save_auto()
                return None
        p = self.plan(recordings)
        try:
            name = self.start(**p["start"], steps=0, hours=self.SESSION_H)
        except (RuntimeError, ValueError) as e:
            log.warning("keep training: %s", e)
            self.auto["error"] = str(e)
            self._save_auto()
            return None
        self.auto.update(sessions=self.auto.get("sessions", 0) + 1, last={"name": name, "start": time.time(), "stage": p["stage"]}, error=None)
        self._save_auto()
        return name

    # ── the guided path: what to train next, in plain words ────

    STAGES = {
        1: ("Practice in the simulator", "Learns to drive around a desk without falling off, in thousands of "
                                         "simulated desks. It can see the exact desk edges here, so this driver only runs in the simulator."),
        2: ("Learn to see", "Learns to do the same using only Vector's camera (with its real lag), by copying the "
                            "simulator driver. This is the driver that runs on your real Vector."),
        3: ("Practice on your desk", "Your own drives on the real desk (recorded in Teleop) teach the camera driver "
                                     "what your desk really looks like. Each session makes it better."),
    }

    def _best(self, pols: list[dict], task: str) -> dict | None:
        c = [p for p in pols if p["task"] == task]
        return max(c, key=lambda p: (bool((p.get("exam") or {}).get("passed")), (p.get("exam") or {}).get("score") or -1e9,
                                     p.get("created") or 0)) if c else None

    def plan(self, recordings: list[dict] | None = None) -> dict[str, Any]:
        """The next thing worth training, and what it builds on. Step 1 until the simulator driver
        passes its test, then the camera driver (step 2), then real drives make it better (step 3)."""
        pols = [p for p in self.policies() if p["ready"] and not p["training"]]
        sim, cam = self._best(pols, "desk"), self._best(pols, "desk-pixels")
        passed = lambda p: bool(p and (p.get("exam") or {}).get("passed"))
        if not passed(sim):
            stage, kw = 1, {"task": "desk", "sensors": "map",
                            "resume": sim["name"] if sim and (self.root / sim["name"] / "model.zip").exists() else None}
        else:
            since = (cam or {}).get("created") or 0
            real = [r["name"] for r in recordings or [] if r.get("frames") and (r.get("created") or 0) > since]
            stage = 3 if passed(cam) else 2
            kw = {"task": "desk-pixels", "sensors": "map", "teacher": sim["name"], "resume": cam["name"] if cam else None, "real": real}
        title, why = self.STAGES[stage]
        return {"stage": stage, "title": title, "why": why, "start": kw,
                "sim": sim and {"name": sim["name"], "passed": passed(sim)},
                "cam": cam and {"name": cam["name"], "passed": passed(cam)},
                "new_drives": len(kw.get("real") or []),
                "continues_from": kw.get("resume")}

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
        self.job, self.started, self.stopping = name, time.time(), False
        return name

    def stop(self) -> None:
        """Returns at once: the run saves and exits on its own (a few seconds); stuck ones get killed."""
        if not self.running() or self.stopping:
            return
        self.stopping = True
        (self.root / self.job / "STOP").touch()
        proc = self.proc

        def reap() -> None:
            try:
                proc.wait(45)
            except subprocess.TimeoutExpired:
                proc.terminate()                         # its last checkpoint is at most a minute old
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        threading.Thread(target=reap, daemon=True).start()

    def last_row(self, name: str) -> dict | None:
        """Just the newest progress line, without reading a marathon run's whole file."""
        try:
            with (self.root / name / "progress.jsonl").open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                lines = [x for x in f.read().splitlines() if x.strip()]
            return json.loads(lines[-1]) if lines else None
        except (OSError, ValueError):
            return None

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
        for d in self.root.iterdir():
            try:
                meta = json.loads((d / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            last = self.last_row(d.name)
            training = d.name == self.running()
            out.append({"name": d.name, **{k: meta.get(k) for k in ("task", "sensors", "steps", "trained_steps", "created", "finished", "hours", "trained_s",
                                                                    "teacher", "built_on", "resumed_from")},
                        "ready": (d / "policy.npz").exists(), "training": training,
                        "stopping": training and self.stopping,
                        "since": self.started if training else None,
                        "can_continue": not meta.get("finished") and not training
                        and ((d / "model.zip").exists() or (d / "student.pt").exists()),
                        "exam": self._json(d / "exam.json"),
                        "last": last, "log_tail": self._tail(d / "train.log")})
        out.sort(key=lambda p: (p["training"], p.get("created") or 0), reverse=True)   # running first, then newest
        return out

    @staticmethod
    def _json(p: Path) -> Any:
        try:
            x = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        return {k: x.get(k) for k in ("passed", "checks", "score", "picked", "picked_steps", "parent", "beats_parent")}

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
