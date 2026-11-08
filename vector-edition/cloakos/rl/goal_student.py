"""The go-to student: drive to a goal from Vector's own camera, learned by copying a teacher (DAgger).

    python -m cloakos.rl.goal_student --out data/policies/goto-pixels-mine --hours 2
    python -m cloakos.rl.goal_student --out ... --teacher data/policies/goto-map-...   # a trained go-to teacher
    python -m cloakos.rl.goal_student --out ... --real data/episodes/<arena go-to recording> ...

Teacher: privileged. In the sim it knows the true pose, the goal and the exact ranges (a trained
go-to driver if you pass --teacher, else a careful point-and-drive controller that stops for edges
and things). On the real desk the Arena go-to (overhead pose + ranges) is the teacher, and its
recorded actions are the labels.

Student: only what Vector has at deploy time. Camera frames (this one + the one before, late and
sometimes dropped, like the real stream), its body sensors (wheel speeds, distance sensor, cliff,
upright, last action) and the goal as it knows it: distance + bearing from its OWN dead-reckoned
pose since the start (wheel odometry, with a little drift), via the same goal_features() the
Arena teacher uses. No overhead information ever reaches it.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .pixelnet import IMG_H, IMG_W
from .student import EXAM_SEEDS, MAX_SPEED, REAL_WEIGHT, LaggyCamera, augment, fold, measured_drop, measured_lag

BODY = 7                      # wheel speeds, distance sensor, cliff, upright, last action (as the desk student)
GOAL = 3                      # goal_features(): distance/3, sin and cos of the bearing
PROP = BODY + GOAL
ODO_SCALE = (0.96, 1.04)      # wheel odometry is a few % off on a real Vector...
ODO_YAW_BIAS = 0.02           # ...and its heading drifts (rad/s, either way)


class DeadReckon:
    """Vector's own idea of where it is: wheel speeds integrated from the start pose, with the
    errors a real one has (per-episode scale per wheel, a slow heading drift)."""

    def __init__(self, rng, start: tuple[float, float, float]) -> None:
        self.x, self.y, self.yaw = start
        k, diff = rng.uniform(*ODO_SCALE), rng.uniform(-0.01, 0.01)    # overall scale, and a small left/right mismatch
        self.kl, self.kr = k * (1 + diff), k * (1 - diff)
        self.bias = float(rng.uniform(-ODO_YAW_BIAS, ODO_YAW_BIAS))

    def step(self, left_mmps: float, right_mmps: float, dt: float) -> dict:
        from ..arena.ranges import predict_pose
        lw, rw = left_mmps * self.kl / 1000, right_mmps * self.kr / 1000
        self.x, self.y, self.yaw = predict_pose(self.x, self.y, self.yaw, lw, rw, dt)
        self.yaw += self.bias * dt
        return {"pos": [self.x, self.y, 0.0], "yaw": self.yaw}


def body(s: dict, last_action) -> list[float]:
    prox = s.get("prox_m")
    return [s["left_mmps"] / MAX_SPEED, s["right_mmps"] / MAX_SPEED, (prox if prox is not None else 0.6) / 0.6,
            1.0 if s.get("cliff") else 0.0, float(s.get("upright", 1.0)), *[float(v) for v in last_action[:2]]]


def scripted_teacher(s: dict, goal, rays) -> np.ndarray:
    """Point-and-drive on the TRUE pose (privileged): turn in place when well off, slow at the end,
    stop and turn away when the exact ranges ahead are short. Forward only (no rear cliff sensors)."""
    ang = math.atan2(goal[1] - s["pos"][1], goal[0] - s["pos"][0]) - s["yaw"]
    ang = math.atan2(math.sin(ang), math.cos(ang))
    dist = math.dist(s["pos"][:2], goal)
    if dist < 0.03:
        return np.zeros(2, np.float32)
    ahead = float(np.min(rays[len(rays) // 2 - 1: len(rays) // 2 + 2])) if rays is not None and len(rays) else 1.0
    if abs(ang) > 0.8 or ahead < 0.06:
        w = 0.25 if (ang > 0 or ahead < 0.06 and ang == 0) else -0.25
        return np.array([-w, w], np.float32)
    v = 0.15 if dist < 0.12 else min(0.45, 0.15 + 2.0 * (dist - 0.12))
    v *= max(0.2, math.cos(ang))
    w = float(np.clip(0.4 * ang, -0.2, 0.2))
    return np.array([v - w, v + w], np.float32)


# ── sim side (worker processes: numpy + MuJoCo only, never torch) ──────────────

def collect(job: tuple) -> dict:
    teacher_path, student_path, beta, steps, seed, stop = job
    from .envs import RAY_MAX, VectorEnv, goal_features
    from .pixelnet import PixelNet, prepare
    from .policy import NumpyPolicy

    env = VectorEnv("goto", "map", seed=seed)
    n_rays = env.spec_.n_rays
    teacher = NumpyPolicy(Path(teacher_path)) if teacher_path else None
    student = PixelNet(student_path) if student_path else None
    rng = np.random.default_rng(seed)
    cam = LaggyCamera(rng, measured_lag(), measured_drop())
    frames, cur, prev, proprio, labels, eps = [], [], [], [], [], []
    obs, _ = env.reset()

    def start_odo():
        s = env.sim.snapshot
        return DeadReckon(rng, (s["pos"][0], s["pos"][1], s["yaw"]))

    odo, est, last = start_odo(), None, np.zeros(2, np.float32)
    n = by_student = 0
    for i in range(steps):
        if i % 200 == 0 and Path(stop).exists():
            break
        s = env.sim.snapshot
        est = odo.step(s["left_mmps"], s["right_mmps"], 1.0 / env.spec_.control_hz) if n else {"pos": [odo.x, odo.y, 0], "yaw": odo.yaw}
        c, pv, new = cam.step(prepare(env.sim.grab(IMG_W, IMG_H)))
        if new:
            frames.append(c)
        ci = len(frames) - 1
        cur.append(ci)
        prev.append(ci - 1 if pv is not c and ci > 0 else ci)
        p = np.asarray(body(s, last) + goal_features(est, env.goal), np.float32)
        proprio.append(p)
        rays = obs[-n_rays:] * RAY_MAX
        label = teacher.predict(obs)[0] if teacher is not None else scripted_teacher(s, env.goal, rays)
        labels.append(label)
        if student is not None and rng.random() >= beta:
            a = student.act(frames[cur[-1]], frames[prev[-1]], p)
            by_student += 1
        else:
            a = label
        last = np.asarray(a, np.float32)
        obs, _, done, trunc, info = env.step(a)
        n += 1
        if done or trunc:
            eps.append({"success": bool(info.get("success")), "fell": bool(info.get("fell")), "bumps": info.get("bumps", 0),
                        "steps": n, "student": by_student / n})
            obs, _ = env.reset()
            cam.reset()
            odo, last, n, by_student = start_odo(), np.zeros(2, np.float32), 0, 0
    return {"frames": np.stack(frames), "cur": np.array(cur, np.int32), "prev": np.array(prev, np.int32),
            "proprio": np.stack(proprio).astype(np.float32), "labels": np.stack(labels).astype(np.float32), "episodes": eps}


def exam(job: tuple) -> dict:
    """The student alone, goals on worlds it has never seen; it only knows its dead-reckoned pose."""
    student_path, seeds = job
    from .envs import VectorEnv, goal_features
    from .pixelnet import PixelNet, prepare

    env = VectorEnv("goto", "map", seed=0, worlds_per_reload=1)
    net = PixelNet(student_path)
    rng = np.random.default_rng(1)
    cam = LaggyCamera(rng, measured_lag(), measured_drop())
    rows = []
    for seed in seeds:
        env.reset(seed=seed)
        cam.reset()
        s = env.sim.snapshot
        odo, last = DeadReckon(rng, (s["pos"][0], s["pos"][1], s["yaw"])), np.zeros(2, np.float32)
        est = {"pos": [odo.x, odo.y, 0], "yaw": odo.yaw}
        steps = bumps = 0
        while True:
            c, pv, _ = cam.step(prepare(env.sim.grab(IMG_W, IMG_H)))
            a = net.act(c, pv, np.asarray(body(s, last) + goal_features(est, env.goal), np.float32))
            _, _, done, trunc, info = env.step(a)
            last = a
            s = env.sim.snapshot
            est = odo.step(s["left_mmps"], s["right_mmps"], 1.0 / env.spec_.control_hz)
            steps += 1
            bumps += bool(info.get("bump"))
            if done or trunc:
                break
        rows.append({"seed": seed, "success": bool(info.get("success")), "fell": bool(info.get("fell")),
                     "bumps": bumps, "seconds": steps / 10})
    return {"runs": rows}


# ── real desk recordings (Arena go-to: the overhead teacher drove) ──────────────

def real_goal_steps(steps: list[dict]) -> list[tuple[dict, list[float], np.ndarray]]:
    """From a recording's steps: (step, goal features from Vector's OWN odometry, teacher label) for
    every Arena go-to step. The goal is moved into the odometry frame once, at the first step of each
    go-to (where the overhead pose and the odometry pose are both known); after that only odometry
    is used, exactly what the student will have on its own."""
    from .envs import goal_features
    out, goal_odo, goal_key = [], None, None
    for s in steps:
        t, ar = s.get("teacher") or {}, s.get("arena") or {}
        pose = (s.get("obs") or {}).get("pose")
        if t.get("task") != "goto" or t.get("action") is None or not pose:
            goal_odo = goal_key = None
            continue
        ox, oy, oyaw = pose[0] / 1000, pose[1] / 1000, float(pose[2])
        if goal_odo is None or goal_key != tuple(t["goal"]):
            if not ar.get("ok") or ar.get("yaw") is None:
                continue                                  # wait for a good overhead fix to anchor the goal
            if t.get("pose_phys") and t.get("goal_phys"):        # v0.9.17+: already un-mirrored
                (ax_, ay_, d), (gx, gy) = t["pose_phys"], t["goal_phys"]
                dx, dy, d = gx - ax_, gy - ay_, float(d)
            else:                                         # older recordings: desk frame (may be mirrored)
                gx, gy = t["goal"]
                dx, dy, d = gx - ar["x"], gy - ar["y"], float(ar["yaw"])
            fwd, left = dx * math.cos(d) + dy * math.sin(d), -dx * math.sin(d) + dy * math.cos(d)   # goal in Vector's frame
            goal_odo = (ox + fwd * math.cos(oyaw) - left * math.sin(oyaw), oy + fwd * math.sin(oyaw) + left * math.cos(oyaw))
            goal_key = tuple(t["goal"])
        feats = goal_features({"pos": [ox, oy], "yaw": oyaw}, goal_odo)
        out.append((s, feats, np.clip(np.asarray(t["action"][:2], np.float32), -1, 1)))
    return out


def load_real(dirs: list[str]) -> dict | None:
    from PIL import Image

    from .pixelnet import prepare
    frames, prev, proprio, labels = [], [], [], []
    for d in map(Path, dirs):
        try:
            steps = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
        except OSError:
            print(f"no recording at {d}", flush=True)
            continue
        last = np.zeros(2, np.float32)
        for s, feats, label in real_goal_steps(steps):
            if not s.get("frame"):
                continue
            try:
                f = prepare(Image.open(d / s["frame"]))
            except OSError:
                continue
            o = s.get("obs") or {}
            prox = o.get("proximity_mm")
            ax, ay, az = o.get("accel") or (0, 0, 9800)
            g = (ax * ax + ay * ay + az * az) ** 0.5 or 1.0
            frames.append(f)
            prev.append(max(0, len(frames) - 2))
            proprio.append([o.get("left_mmps", 0) / MAX_SPEED, o.get("right_mmps", 0) / MAX_SPEED,
                            (prox / 1000 if prox else 0.6) / 0.6, 1.0 if o.get("cliff") else 0.0, az / g, *last, *feats])
            labels.append(label)
            last = label
    if not frames:
        return None
    return {"frames": np.stack(frames), "prev": np.array(prev, np.int32), "proprio": np.asarray(proprio, np.float32),
            "labels": np.stack(labels)}


# ── the network ─────────────────────────────────────────────────

def build(torch, prop: int = PROP):
    """The desk student's network with a wider sensor input (body + goal); PixelNet runs it as is."""
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ch = [6, 16, 32, 48, 64]
            self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 5, 2, 2) for i in range(4)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(ch[i + 1]) for i in range(4)])
            self.fc0 = nn.Linear(64 * 4 * 6 + prop, 128)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 2)
            self.drop = nn.Dropout(0.1)

        def forward(self, img, p):
            x = img - 0.5
            for c, b in zip(self.convs, self.bns):
                x = torch.relu(b(c(x)))
            x = self.drop(torch.relu(self.fc0(torch.cat([x.flatten(1), p], 1))))
            return torch.tanh(self.fc2(torch.relu(self.fc1(x))))

    return Net()


def batches(parts: list[tuple[dict, int]], rng, size: int):
    index = np.concatenate([np.stack([np.full(len(p["labels"]), k), np.arange(len(p["labels"]))], 1).repeat(w, 0)
                            for k, (p, w) in enumerate(parts)])
    rng.shuffle(index)
    for s in range(0, len(index) - size + 1, size):
        img = np.empty((size, 6, IMG_H, IMG_W), np.uint8)
        prop = np.empty((size, PROP), np.float32)
        lab = np.empty((size, 2), np.float32)
        for j, (k, i) in enumerate(index[s:s + size]):
            p = parts[k][0]
            img[j, :3] = p["frames"][p["cur"][i] if "cur" in p else i]
            img[j, 3:] = p["frames"][p["prev"][i]]
            prop[j], lab[j] = p["proprio"][i], p["labels"][i]
        yield img, prop, lab


def train_round(torch, net, opt, parts, rng, epochs: int = 3, size: int = 256, stop: Path | None = None) -> list[float]:
    net.train()
    losses = []
    for _ in range(epochs):
        for img, prop, lab in batches(parts, rng, size):
            if stop is not None and stop.exists():
                return losses
            x, pp = augment(torch, torch.from_numpy(img).float() / 255.0, torch.from_numpy(prop))
            pp = pp.clone()
            pp[:, BODY:] = torch.from_numpy(prop[:, BODY:])          # the goal isn't sensor noise: keep it exact
            loss = torch.nn.functional.mse_loss(net(x, pp), torch.from_numpy(lab))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    net.eval()
    return losses


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cloakos.rl.goal_student")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default=None, help="a trained go-to policy folder (default: the built-in privileged controller)")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--envs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--real", nargs="*", default=[], help="real-desk recordings of Arena go-to runs")
    ap.add_argument("--max-samples", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=None)
    a = ap.parse_args(argv)
    import torch

    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    if a.hours:
        a.steps, a.rounds = 25_000 * 10_000, 10_000
    a.rounds = a.rounds or int(np.clip(a.steps // 25_000, 2, 400))
    teacher = None
    if a.teacher:
        tm = json.loads((Path(a.teacher) / "meta.json").read_text())
        if tm.get("task") != "goto" or tm.get("sensors") != "map":
            raise SystemExit(f"{Path(a.teacher).name} isn't a go-to teacher with map sensors")
        teacher = str(Path(a.teacher) / "policy.npz")
    meta = {"task": "goto-pixels", "sensors": "pixels", "obs_size": PROP, "control_hz": 10.0, "max_speed": MAX_SPEED,
            "steps": a.steps, "envs": a.envs, "created": time.time(), "algo": "DAgger",
            "teacher": Path(a.teacher).name if a.teacher else "privileged controller",
            "goal_input": "goal_features from dead-reckoned odometry", "real": [Path(r).name for r in a.real]}
    if a.hours:
        meta["hours"] = a.hours
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    net = build(torch)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    parts: list[tuple[dict, int]] = []
    if a.real:
        part = load_real(a.real)
        if part:
            parts.append((part, REAL_WEIGHT))
            print(f"real desk: {len(part['labels'])} go-to steps", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    progress = (out / "progress.jsonl").open("a")
    rng = np.random.default_rng(a.seed)
    per_round = max(500, a.steps // a.rounds)
    t0, total, budget = time.time(), 0, (a.hours or 0) * 3600
    stop = out / "STOP"
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.envs) as pool:
        for k in range(a.rounds):
            if stop.exists() or (budget and time.time() - t0 >= budget):
                break
            beta = 1.0 if k == 0 else 0.5 ** k
            student = str(out / "policy.npz") if (out / "policy.npz").exists() else None
            jobs = [(teacher, student, beta, per_round // a.envs, a.seed * 1000 + k * 100 + i, str(stop)) for i in range(a.envs)]
            res = pool.map(collect, jobs)
            eps = [e for p in res for e in p["episodes"]]
            for p in res:
                parts.append(({kk: v for kk, v in p.items() if kk != "episodes"}, 1))
            while sum(len(p["labels"]) for p, _ in parts) > a.max_samples:
                i = next((j for j, (_, w) in enumerate(parts) if w == 1), None)
                if i is None or len(parts) == 1:
                    break
                parts.pop(i)
            total += sum(len(p["labels"]) for p in res)
            losses = train_round(torch, net, opt, parts, rng, stop=stop)
            torch.save(net.state_dict(), out / "student.pt")
            np.savez(out / "policy.tmp.npz", **fold(net))
            (out / "policy.tmp.npz").replace(out / "policy.npz")
            driven = [e for e in eps if e["student"] > 0.5] or eps
            row = {"t": round(time.time() - t0, 1), "steps": total, "round": k + 1, "teacher_share": round(beta, 3),
                   "episodes": len(eps), "loss": round(float(np.mean(losses[-50:])), 4) if losses else None,
                   "success": round(float(np.mean([e["success"] for e in driven])), 3) if driven else None,
                   "fell": round(float(np.mean([e["fell"] for e in driven])), 3) if driven else None,
                   "reward": round(float(np.mean([e["success"] for e in driven])), 3) if driven else None}
            if budget:
                row["done"] = round(min(1.0, (time.time() - t0) / budget), 4)
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            print(json.dumps(row), flush=True)
            meta.update(rounds=k + 1, trained_steps=total)
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
        progress.close()
        stop.unlink(missing_ok=True)
        if not (out / "policy.npz").exists():
            return
        meta["finished"] = time.time()
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print("taking the exam", flush=True)
        seeds = list(EXAM_SEEDS)
        res = pool.map(exam, [(str(out / "policy.npz"), seeds[i::a.envs]) for i in range(a.envs) if seeds[i::a.envs]])
    from .evaluate import summarize
    ex = {**summarize("goto", "pixels", [r for x in res for r in x["runs"]]), "task": "goto-pixels"}
    (out / "exam.json").write_text(json.dumps(ex, indent=1))
    print(json.dumps({k: v for k, v in ex.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
