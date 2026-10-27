"""TTS engines. Each returns (float32 mono samples in -1..1, sample_rate). Pick with voice.tts.engine."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class TTS(Protocol):
    id: str   # engine+voice; part of the cache key

    def synth(self, text: str) -> tuple[np.ndarray, int]: ...


def make_tts(cfg: dict) -> TTS:
    c = (cfg.get("voice", {}) or {}).get("tts", {}) or {}
    engine = c.get("engine", "kokoro")
    if engine == "kokoro":
        from .kokoro import KokoroTTS
        return KokoroTTS(c.get("kokoro", {}) or {})
    if engine == "piper":
        from .piper import PiperTTS
        return PiperTTS(c.get("piper", {}) or {})
    raise ValueError(f"unknown TTS engine: {engine}")
