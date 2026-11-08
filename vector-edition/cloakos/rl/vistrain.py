"""Train the VisionNet on pictures from cloakos.rl.visdata (needs torch).

    python -m cloakos.rl.vistrain data/vision/part*.npz --out data/vision/visionnet.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np


def build(torch):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ch = [3, 16, 32, 48, 64]
            self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 5, 2, 2) for i in range(4)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(ch[i + 1]) for i in range(4)])
            self.fc0 = nn.Linear(64 * 4 * 6, 128)
            self.fc1 = nn.Linear(128, 11)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            x = x - 0.5
            for c, b in zip(self.convs, self.bns):
                x = torch.relu(b(c(x)))
            x = self.drop(torch.relu(self.fc0(x.flatten(1))))
            return self.fc1(x)

    return Net()


def augment(torch, x):
    """Camera variation the sim doesn't cover: brightness, contrast, colour cast, grain, small shifts."""
    n = x.shape[0]
    x = x * (0.6 + 0.8 * torch.rand(n, 1, 1, 1)) * (0.92 + 0.16 * torch.rand(n, 3, 1, 1))
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * (0.7 + 0.6 * torch.rand(n, 1, 1, 1)) + m
    x = x + torch.randn_like(x) * (0.01 + 0.04 * torch.rand(n, 1, 1, 1))
    dx = int(torch.randint(-3, 4, (1,)))
    return torch.clamp(torch.roll(x, dx, dims=3), 0, 1)


def fold(net) -> dict:
    """Fold batch-norm into the convolutions so numpy inference needs nothing else."""
    out: dict = {"n_conv": np.array(len(net.convs))}
    for i, (c, b) in enumerate(zip(net.convs, net.bns)):
        s = (b.weight / (b.running_var + b.eps).sqrt()).detach()
        out[f"w{i}"] = (c.weight.detach() * s[:, None, None, None]).numpy().astype(np.float32)
        out[f"b{i}"] = ((c.bias.detach() - b.running_mean) * s + b.bias.detach()).numpy().astype(np.float32)
    out["fw0"], out["fb0"] = net.fc0.weight.detach().numpy(), net.fc0.bias.detach().numpy()
    out["fw1"], out["fb1"] = net.fc1.weight.detach().numpy(), net.fc1.bias.detach().numpy()
    return out


def main(argv: list[str] | None = None) -> None:
    import torch

    ap = argparse.ArgumentParser(prog="cloakos.rl.vistrain")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default="data/vision/visionnet.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    xs, ys, vx, vy = [], [], [], []
    for f in sorted(p for pat in a.data for p in glob.glob(pat)):
        d = np.load(f)
        cut = int(len(d["x"]) * 0.9)                 # the last 10% of every file are rooms the net never trains on
        xs.append(d["x"][:cut]); ys.append(d["y"][:cut]); vx.append(d["x"][cut:]); vy.append(d["y"][cut:])
    X = torch.from_numpy(np.concatenate(xs)).permute(0, 3, 1, 2).contiguous()
    Y = torch.from_numpy(np.concatenate(ys).astype(np.float32))
    VX = torch.from_numpy(np.concatenate(vx)).permute(0, 3, 1, 2).float() / 255
    VY = torch.from_numpy(np.concatenate(vy).astype(np.float32))
    print(f"{len(X)} train, {len(VX)} held-out frames", flush=True)
    net = build(torch)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=a.epochs * ((len(X) + 127) // 128))
    base = float((VY - Y.mean(0)).abs().mean())
    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            x = augment(torch, X[idx].float() / 255)
            y = Y[idx]
            p = torch.sigmoid(net(x))
            w = 1.0 + 2.0 * (1.0 - y)                  # close things matter most
            loss = (w * (p - y).abs()).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(VX))
            mae = float((p - VY).abs().mean())
            near = VY < 0.5
            nmae = float((p - VY).abs()[near].mean()) if near.any() else None
            miss = float(((p > 0.8) & (VY < 0.3)).float().sum() / max(1, (VY < 0.3).sum()))   # said clear, it wasn't
        hist.append({"epoch": ep, "mae_m": mae, "near_mae_m": nmae, "missed_close": miss})
        print(f"epoch {ep + 1}/{a.epochs}  mae {mae:.3f} m  near {nmae:.3f} m  missed-close {miss:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out = fold(net)
    out["m_baseline_mae"] = np.array(base)
    out["m_mae"] = np.array(hist[-1]["mae_m"]); out["m_near_mae"] = np.array(hist[-1]["near_mae_m"] or 0.0)
    out["m_missed_close"] = np.array(hist[-1]["missed_close"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    Path(a.out).with_suffix(".json").write_text(json.dumps({"baseline_mae_m": base, **hist[-1], "train_frames": len(X), "heldout_frames": len(VX)}, indent=1))
    # the numpy runner has to agree with torch
    from ..mapping.visionnet import VisionNet
    vn = VisionNet(a.out)
    with torch.no_grad():
        ref = torch.sigmoid(net(VX[:8])).numpy()
    got = np.stack([vn.predict(VX[i].numpy().astype(np.float32)) for i in range(8)])
    print("numpy vs torch max diff:", float(np.abs(ref - got).max()), flush=True)


if __name__ == "__main__":
    main()
