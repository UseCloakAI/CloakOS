"""Teach the desk student to drive from pixels by copying the desk teacher (DAgger).

    python -m cloakos.rl.student --teacher data/policies/desk-map-... --out data/policies/desk-pixels-... --steps 200000

Each round the sims drive (first the teacher, then more and more the student), the teacher says
what it would have done at every step, and the student learns from all of it so far. The student
only sees what the real Vector has: camera frames and its own body sensors.

Stacking:  --init <earlier student>  keeps its weights and everything it has already seen.
Real Vector: --real data/episodes/<recording> ...  adds your own drives on the real desk (camera +
what you did with teleop) as extra lessons, weighted up, so each real session makes it more accurate.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .pixelnet import IMG_H, IMG_W, PROPRIO

MAX_SPEED = 150.0
REAL_WEIGHT = 3          # a real-desk sample counts this many times
EXAM_SEEDS = range(900_001, 900_021)
CAM_LAG_STEPS = (2, 5)   # Vector's camera over Wi-Fi: a frame is 0.2-0.5 s old when the brain gets it (until measured)
CAM_DROP = 0.15          # and sometimes the next one doesn't make it in time


def measured_drop(path: str = "data/calibration.json") -> float:
    """Share of 0.1 s ticks with no new frame: from the Camera lag calibration's frame rate (5 fps -> half)."""
    try:
        fps = float(json.loads(Path(path).read_text())["camera"]["frames_per_s"])
        return float(np.clip(1 - fps / 10, 0.0, 0.9))
    except (OSError, ValueError, KeyError, TypeError):
        return CAM_DROP


def measured_lag(path: str = "data/calibration.json") -> tuple[int, int]:
    """Use the Camera lag calibration when there is one: its delay (in 0.1 s steps), give or take one."""
    try:
        c = json.loads(Path(path).read_text()).get("camera") or {}
        k = int(round(float(c["picture_behind_sensors_s"]) * 10))
        return max(0, k - 1), max(1, k + 1)
    except (OSError, ValueError, KeyError, TypeError):
        return CAM_LAG_STEPS


class LaggyCamera:
    """The real stream as the brain sees it: late, sometimes stuck on the last frame."""

    def __init__(self, rng, lag: tuple[int, int] = CAM_LAG_STEPS, drop: float = CAM_DROP) -> None:
        self.rng, self.lag_range, self.drop = rng, lag, drop
        self.reset()

    def reset(self) -> None:
        self.lag = int(self.rng.integers(self.lag_range[0], self.lag_range[1] + 1))
        self.shots: list = []
        self.cur = self.prev = None

    def step(self, img) -> tuple:
        """Feed this tick's render; returns (newest frame delivered, the one before it, is it new)."""
        self.shots.append(img)
        self.shots = self.shots[-(self.lag + 1):]
        new = self.cur is None or self.rng.random() >= self.drop
        if new:
            self.prev, self.cur = self.cur, self.shots[0]
        return self.cur, self.prev if self.prev is not None else self.cur, new


# ── sim side (worker processes: numpy + MuJoCo only, never torch) ──────────────

def collect(job: tuple) -> dict:
    teacher_path, student_path, beta, steps, seed, stop = job
    from .envs import VectorEnv
    from .pixelnet import PixelNet, prepare
    from .policy import NumpyPolicy

    env = VectorEnv("desk", "map", seed=seed)
    teacher = NumpyPolicy(Path(teacher_path))
    student = PixelNet(student_path) if student_path else None
    rng = np.random.default_rng(seed)
    cam = LaggyCamera(rng, measured_lag(), measured_drop())
    frames, cur, prev, proprio, labels, eps = [], [], [], [], [], []
    obs, _ = env.reset()
    n = by_student = 0
    for i in range(steps):
        if i % 200 == 0 and Path(stop).exists():        # Stop pressed: hand back what's done so far
            break
        c, pv, new = cam.step(prepare(env.sim.grab(IMG_W, IMG_H)))
        if new:                                          # store each delivered frame once
            frames.append(c)
        ci = len(frames) - 1
        cur.append(ci)
        prev.append(ci - 1 if pv is not c and ci > 0 else ci)
        p = obs[:PROPRIO].copy()
        proprio.append(p)
        label, _ = teacher.predict(obs)                  # what the teacher would do here, always
        labels.append(label)
        if student is not None and rng.random() >= beta:
            a = student.act(frames[cur[-1]], frames[prev[-1]], p)
            by_student += 1
        else:
            a = label
        obs, _, done, trunc, info = env.step(a)
        n += 1
        if done or trunc:
            eps.append({"cells": info.get("cells", 0), "fell": bool(info.get("fell")), "bumps": info.get("bumps", 0),
                        "steps": n, "student": by_student / n})
            obs, _ = env.reset()
            cam.reset()
            n = by_student = 0
    return {"frames": np.stack(frames), "cur": np.array(cur, np.int32), "prev": np.array(prev, np.int32), "proprio": np.stack(proprio).astype(np.float32),
            "labels": np.stack(labels).astype(np.float32), "episodes": eps}


def exam(job: tuple) -> dict:
    """The student alone, on desks it has never seen."""
    student_path, seeds = job
    from .envs import VectorEnv
    from .pixelnet import PixelNet, prepare

    env = VectorEnv("desk", "map", seed=0, worlds_per_reload=1)
    net = PixelNet(student_path)
    cam = LaggyCamera(np.random.default_rng(1), measured_lag(), measured_drop())
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        cam.reset()
        steps = bumps = 0
        while True:
            cur, last, _ = cam.step(prepare(env.sim.grab(IMG_W, IMG_H)))
            obs, _, done, trunc, info = env.step(net.act(cur, last, obs[:PROPRIO]))
            steps += 1
            bumps += bool(info.get("bump"))
            if done or trunc:
                break
        rows.append({"seed": seed, "fell": bool(info.get("fell")), "bumps": bumps, "cells": info.get("cells", 0),
                     "minutes": steps / 600})
    return {"runs": rows}


# ── real desk recordings ────────────────────────────────────────

def load_real(dirs: list[str]) -> dict | None:
    """Steps with a camera frame from cloakos recordings: what you did with teleop is the label, or,
    under the Arena, what a teacher driver (steering from the overhead camera's ranges) did."""
    from PIL import Image

    from .pixelnet import prepare

    frames, prev, proprio, labels = [], [], [], []
    for d in map(Path, dirs):
        last_frame, last_act = None, np.zeros(2, np.float32)
        try:
            lines = (d / "steps.jsonl").read_text().splitlines()
        except OSError:
            print(f"no recording at {d}", flush=True)
            continue
        for line in lines:
            s = json.loads(line)
            act = s.get("action") or {}
            a = np.clip([act.get("left_mmps", 0) / MAX_SPEED, act.get("right_mmps", 0) / MAX_SPEED], -1, 1).astype(np.float32)
            t = s.get("teacher") or {}
            if t.get("action") is not None and s.get("action_source") == "learned":
                a = np.clip(np.asarray(t["action"][:2], np.float32), -1, 1)
            if s.get("frame") and (s.get("action_source") == "teleop" or t.get("action") is not None):
                try:
                    f = prepare(Image.open(d / s["frame"]))
                except OSError:
                    continue
                o = s.get("obs") or {}
                prox = o.get("proximity_mm")
                ax, ay, az = o.get("accel") or (0, 0, 9800)
                g = (ax * ax + ay * ay + az * az) ** 0.5 or 1.0
                frames.append(f)
                prev.append(len(frames) - 2 if last_frame is not None else len(frames) - 1)
                proprio.append([o.get("left_mmps", 0) / MAX_SPEED, o.get("right_mmps", 0) / MAX_SPEED,
                                (prox / 1000 if prox else 0.6) / 0.6, 1.0 if o.get("cliff") else 0.0, az / g, *last_act])
                labels.append(a)
                last_frame = f
            elif s.get("frame"):
                last_frame = None
            last_act = a
    if not frames:
        return None
    return {"frames": np.stack(frames), "prev": np.array(prev, np.int32), "proprio": np.asarray(proprio, np.float32),
            "labels": np.stack(labels)}


# ── the network (training side) ─────────────────────────────────

def build(torch):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ch = [6, 16, 32, 48, 64]
            self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 5, 2, 2) for i in range(4)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(ch[i + 1]) for i in range(4)])
            self.fc0 = nn.Linear(64 * 4 * 6 + PROPRIO, 128)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 2)
            self.drop = nn.Dropout(0.1)

        def forward(self, img, prop):
            x = img - 0.5
            for c, b in zip(self.convs, self.bns):
                x = torch.relu(b(c(x)))
            x = self.drop(torch.relu(self.fc0(torch.cat([x.flatten(1), prop], 1))))
            return torch.tanh(self.fc2(torch.relu(self.fc1(x))))

    return Net()


def augment(torch, x, p):
    """Camera and sensor variation the sim doesn't cover, the same on both frames of a sample."""
    n = x.shape[0]
    x = x.view(n, 2, 3, IMG_H, IMG_W)
    x = x * (0.5 + 1.0 * torch.rand(n, 1, 1, 1, 1)) * (0.9 + 0.2 * torch.rand(n, 1, 3, 1, 1))
    m = x.mean((2, 3, 4), keepdim=True)
    x = (x - m) * (0.7 + 0.6 * torch.rand(n, 1, 1, 1, 1)) + m
    x = x + torch.randn_like(x) * (0.01 + 0.04 * torch.rand(n, 1, 1, 1, 1))
    x = torch.roll(x, int(torch.randint(-3, 4, (1,))), dims=4)
    p = p + torch.randn_like(p) * 0.03
    return torch.clamp(x, 0, 1).view(n, 6, IMG_H, IMG_W), p


def fold(net) -> dict:
    out: dict = {"n_conv": np.array(len(net.convs))}
    for i, (c, b) in enumerate(zip(net.convs, net.bns)):
        s = (b.weight / (b.running_var + b.eps).sqrt()).detach()
        out[f"w{i}"] = (c.weight.detach() * s[:, None, None, None]).numpy().astype(np.float32)
        out[f"b{i}"] = ((c.bias.detach() - b.running_mean) * s + b.bias.detach()).numpy().astype(np.float32)
    for i, fc in enumerate((net.fc0, net.fc1, net.fc2)):
        out[f"fw{i}"], out[f"fb{i}"] = fc.weight.detach().numpy(), fc.bias.detach().numpy()
    return out


class Data:
    """Every lesson so far, in memory as uint8 frames."""

    def __init__(self, max_samples: int) -> None:
        self.parts: list[tuple[dict, int]] = []
        self.max = max_samples

    def add(self, part: dict, weight: int = 1) -> None:
        self.parts.append((part, weight))
        while sum(len(p["labels"]) for p, _ in self.parts) > self.max:
            sim = next((k for k, (_, w) in enumerate(self.parts) if w == 1), None)
            if sim is None or len(self.parts) == 1:
                break
            self.parts.pop(sim)                                       # forget the oldest sim lessons first, never real ones

    def __len__(self) -> int:
        return sum(len(p["labels"]) * w for p, w in self.parts)

    def batches(self, rng, size: int):
        index = np.concatenate([np.stack([np.full(len(p["labels"]), k), np.arange(len(p["labels"]))], 1).repeat(w, 0)
                                for k, (p, w) in enumerate(self.parts)])
        rng.shuffle(index)
        for s in range(0, len(index) - size + 1, size):
            b = index[s:s + size]
            img = np.empty((size, 6, IMG_H, IMG_W), np.uint8)
            prop = np.empty((size, PROPRIO), np.float32)
            lab = np.empty((size, 2), np.float32)
            for j, (k, i) in enumerate(b):
                p = self.parts[k][0]
                img[j, :3] = p["frames"][p["cur"][i] if "cur" in p else i]
                img[j, 3:] = p["frames"][p["prev"][i]]
                prop[j], lab[j] = p["proprio"][i], p["labels"][i]
            yield img, prop, lab


def save_part(part: dict, path: Path) -> None:
    np.savez_compressed(path, **{k: v for k, v in part.items() if k != "episodes"})


def load_part(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cloakos.rl.student")
    ap.add_argument("--teacher", required=True, help="a trained desk policy folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200_000, help="sim steps in total, over all rounds")
    ap.add_argument("--rounds", type=int, default=None, help="default: one per 25k steps (4-400)")
    ap.add_argument("--continue", dest="cont", action="store_true", help="pick a stopped/crashed run in --out back up")
    ap.add_argument("--envs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--init", default=None, help="an earlier student to build on")
    ap.add_argument("--real", nargs="*", default=[], help="recordings from the real desk (teleop)")
    ap.add_argument("--max-samples", type=int, default=150_000, help="memory cap (~15 KB each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=None, help="train for this long instead of a step count (then test)")
    a = ap.parse_args(argv)

    import torch

    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    if a.hours:                                           # a time budget: 25k-step rounds until the time's up
        a.steps, a.rounds = 25_000 * 10_000, 10_000
    a.rounds = a.rounds or int(np.clip(a.steps // 25_000, 4, 400))
    teacher = Path(a.teacher)
    tmeta = json.loads((teacher / "meta.json").read_text())
    if tmeta.get("task") != "desk":
        raise SystemExit(f"{teacher.name} isn't a desk teacher (it's {tmeta.get('task')})")
    meta = {"task": "desk-pixels", "sensors": "pixels", "obs_size": PROPRIO, "control_hz": 10.0, "max_speed": MAX_SPEED,
            "steps": a.steps, "envs": a.envs, "created": time.time(), "algo": "DAgger", "teacher": teacher.name,
            "built_on": None, "real": [Path(r).name for r in a.real], "data_dirs": [str(out / "data")]}
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    net = build(torch)
    data = Data(a.max_samples)
    rounds_done = 0
    prev_meta = json.loads((out / "meta.json").read_text()) if a.cont and (out / "meta.json").exists() else None
    if prev_meta and (out / "student.pt").exists():    # crashed or stopped: same folder, same plan, carry on
        meta = {**prev_meta, "continued": prev_meta.get("continued", 0) + 1}
        meta.pop("finished", None)
        net.load_state_dict(torch.load(out / "student.pt"))
        rounds_done, a.rounds, a.steps = int(meta.get("rounds", 0)), int(meta["rounds_total"]), int(meta["steps"])
        for d in meta.get("data_dirs", []):
            for f in sorted(Path(d).glob("*.npz")):
                data.add(load_part(f), REAL_WEIGHT if f.name.startswith("real") else 1)
        print(f"continuing at round {rounds_done + 1} of {a.rounds}: {len(data)} lessons", flush=True)
    elif a.init:                                          # stack on an earlier student: its weights + its lessons
        init = Path(a.init)
        imeta = json.loads((init / "meta.json").read_text())
        net.load_state_dict(torch.load(init / "student.pt"))
        meta["built_on"] = init.name
        rounds_done = int(imeta.get("rounds", 0))
        a.rounds += rounds_done
        meta["data_dirs"] = imeta.get("data_dirs", []) + meta["data_dirs"]
        for d in imeta.get("data_dirs", []):
            for f in sorted(Path(d).glob("*.npz")):
                data.add(load_part(f), REAL_WEIGHT if f.name.startswith("real") else 1)
        print(f"building on {init.name}: {len(data)} lessons", flush=True)
    if a.real and not prev_meta:
        part = load_real(a.real)
        if part:
            save_part(part, out / "data" / f"real_{int(time.time())}.npz")
            data.add(part, REAL_WEIGHT)
            print(f"real desk: {len(part['labels'])} teleop steps", flush=True)
    meta.setdefault("rounds", rounds_done)
    meta["rounds_total"] = a.rounds
    if a.hours and not prev_meta:
        meta["hours"] = a.hours
    budget = float(meta.get("hours") or 0) * 3600
    t_run, trained_s, hit = time.time(), float(meta.get("trained_s", 0.0)), False
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    progress = (out / "progress.jsonl").open("a")
    opt = torch.optim.Adam(net.parameters(), lr=3e-4 if (a.init or prev_meta) else 1e-3)
    rng = np.random.default_rng(a.seed)
    per_round = int(meta.get("per_round") or max(500, a.steps // max(1, a.rounds - rounds_done)))
    meta["per_round"] = per_round
    t0, total = time.time(), int(meta.get("trained_steps", 0)) if prev_meta else 0
    ctx = mp.get_context("spawn")                        # workers never inherit torch (it and offscreen GL don't mix)
    with ctx.Pool(a.envs) as pool:
        for k in range(rounds_done, a.rounds):
            if (out / "STOP").exists():
                break
            if budget and trained_s + time.time() - t_run >= budget:     # time's up: that counts as done
                hit = True
                break
            beta = 1.0 if k == 0 else 0.5 ** k            # who drives: the teacher at first, the student later
            student = str(out / "policy.npz") if (out / "policy.npz").exists() else None
            if k > 0 and student is None and a.init:
                student = str(Path(a.init) / "policy.npz")
            print(f"round {k + 1} of {a.rounds}: driving the desk", flush=True)
            jobs = [(str(teacher / "policy.npz"), student, beta, per_round // a.envs, a.seed * 1000 + k * 100 + i,
                     str(out / "STOP")) for i in range(a.envs)]
            parts = pool.map(collect, jobs)
            eps = [e for p in parts for e in p["episodes"]]
            for i, p in enumerate(parts):
                save_part(p, out / "data" / f"r{k:03d}_{i}.npz")
                data.add({kk: v for kk, v in p.items() if kk != "episodes"})
            total += sum(len(p["labels"]) for p in parts)
            if (out / "STOP").exists():
                break
            print(f"round {k + 1} of {a.rounds}: learning from {len(data)} lessons", flush=True)
            net.train()
            losses = []
            for _ in range(3):
                if (out / "STOP").exists():
                    break
                for img, prop, lab in data.batches(rng, 256):
                    x, pp = augment(torch, torch.from_numpy(img).float() / 255.0, torch.from_numpy(prop))
                    loss = torch.nn.functional.mse_loss(net(x, pp), torch.from_numpy(lab))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
            net.eval()
            torch.save(net.state_dict(), out / "student.tmp.pt")
            (out / "student.tmp.pt").replace(out / "student.pt")
            z = fold(net)
            np.savez(out / "policy.tmp.npz", **z)
            (out / "policy.tmp.npz").replace(out / "policy.npz")
            driven = [e for e in eps if e["student"] > 0.5] or eps
            row = {"t": round(time.time() - t0, 1), "steps": total, "round": k + 1, "teacher_share": round(beta, 3),
                   "episodes": len(eps), "lessons": len(data), "loss": round(float(np.mean(losses[-50:])), 4) if losses else None,
                   "reward": round(float(np.mean([e["cells"] for e in driven])), 2) if driven else None,
                   "success": round(1 - float(np.mean([e["fell"] for e in driven])), 3) if driven else None,
                   "fell": round(float(np.mean([e["fell"] for e in driven])), 3) if driven else None,
                   "bumps": round(float(np.mean([e["bumps"] for e in driven])), 2) if driven else None,
                   "fps": int(total / max(1e-6, time.time() - t0))}
            if budget:
                row["done"] = round(min(1.0, (trained_s + time.time() - t_run) / budget), 4)
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            print(json.dumps(row), flush=True)
            meta.update(rounds=k + 1, trained_steps=total, lessons=len(data))      # a checkpoint every round
            if budget:
                meta["trained_s"] = trained_s + time.time() - t_run
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
        progress.close()
        (out / "STOP").unlink(missing_ok=True)
        if (meta["rounds"] < a.rounds and not hit) or not (out / "policy.npz").exists():
            return                                       # stopped early: left open to Continue
        meta["finished"] = time.time()
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print("taking the exam", flush=True)
        seeds = list(EXAM_SEEDS)
        res = pool.map(exam, [(str(out / "policy.npz"), seeds[i::a.envs]) for i in range(a.envs) if seeds[i::a.envs]])
    write_exam(out, [row for x in res for row in x["runs"]])
    from .. import share
    if share.enabled():
        try:
            share.upload_run(out, os.environ.get("CLOAK_SHARE_SOURCE", "local"))
        except Exception as e:
            print(f"couldn't share this run: {e}", flush=True)


def write_exam(out: Path, rows: list[dict]) -> dict:
    from .evaluate import summarize

    res = {**summarize("desk", "pixels", rows), "task": "desk-pixels"}
    (out / "exam.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}), flush=True)
    return res


if __name__ == "__main__":
    main()
