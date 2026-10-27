"""Creature chirps, synthesized: no samples, no Vector sound bank. 16 kHz mono int16 PCM.

Each chirp is a few notes; a note glides from f0 to f1 with a soft attack/release and a
touch of 2nd harmonic so it reads as a voice, not a beep.
"""
from __future__ import annotations

import numpy as np

SR = 16000

# (f0 Hz, f1 Hz, seconds, gap after)
CHIRPS: dict[str, list[tuple[float, float, float, float]]] = {
    "happy":   [(900, 1400, 0.09, 0.03), (1100, 1700, 0.11, 0.0)],
    "greet":   [(700, 900, 0.08, 0.02), (900, 1200, 0.08, 0.02), (1200, 1600, 0.14, 0.0)],
    "curious": [(800, 760, 0.10, 0.02), (760, 1500, 0.20, 0.0)],
    "scared":  [(1900, 900, 0.06, 0.0), (1700, 800, 0.06, 0.0), (1500, 600, 0.10, 0.0)],
    "sad":     [(700, 520, 0.28, 0.05), (560, 380, 0.35, 0.0)],
    "sleepy":  [(420, 600, 0.25, 0.0), (600, 300, 0.45, 0.0)],
    "bored":   [(520, 520, 0.18, 0.08), (470, 420, 0.25, 0.0)],
    "startle": [(1200, 2200, 0.05, 0.0), (2200, 1600, 0.07, 0.0)],
    "wake":    [(350, 700, 0.20, 0.03), (900, 1200, 0.10, 0.0)],
}


def note(f0: float, f1: float, dur: float, sr: int = SR) -> np.ndarray:
    n = max(1, int(dur * sr))
    t = np.linspace(0, 1, n, endpoint=False)
    freq = f0 * (f1 / f0) ** t                      # exponential glide sounds natural
    phase = 2 * np.pi * np.cumsum(freq) / sr
    wave = np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
    a = max(1, int(0.012 * sr))
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    env[-a * 3:] *= np.linspace(1, 0, min(n, a * 3))
    return (wave * env).astype(np.float32)


def chirp(name: str, pitch: float = 1.0, gain: float = 0.6) -> bytes:
    parts = []
    for f0, f1, dur, gap in CHIRPS[name]:
        parts.append(note(f0 * pitch, f1 * pitch, dur))
        if gap:
            parts.append(np.zeros(int(gap * SR), np.float32))
    x = np.concatenate(parts)
    x = x / max(1e-6, float(np.max(np.abs(x)))) * gain
    return (x * 32767).astype("<i2").tobytes()
