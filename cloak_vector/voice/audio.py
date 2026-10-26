"""Audio prep for Vector's speaker: 16 kHz, 16-bit, mono WAV.

`pitch` > 1 plays the clip faster and higher (cheap creature voice): we resample to
16000/pitch and Vector plays it back at 16000.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

VECTOR_SR = 16000


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        n = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def to_vector_wav(samples: np.ndarray, sr: int, path: str | Path, pitch: float = 1.0, gain: float = 0.9) -> Path:
    x = np.asarray(samples, dtype=np.float32).flatten()
    x = resample(x, sr, int(round(VECTOR_SR / max(pitch, 0.25))))
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0:
        x = x / peak * gain
    pad = np.zeros(int(VECTOR_SR * 0.05), np.float32)   # 50 ms tail so the last syllable isn't clipped
    pcm = (np.concatenate([x, pad]) * 32767).astype(np.int16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(VECTOR_SR)
        w.writeframes(pcm.tobytes())
    return path
