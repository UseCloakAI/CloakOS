"""Kokoro via kokoro-onnx (no torch). pip install .[kokoro], then grab the model files (see README)."""
from __future__ import annotations

import numpy as np


class KokoroTTS:
    def __init__(self, c: dict) -> None:
        from kokoro_onnx import Kokoro

        self.voice = c.get("voice", "af_heart")
        self.speed = float(c.get("speed", 1.1))
        self.lang = c.get("lang", "en-us")
        self.id = f"kokoro:{self.voice}:{self.speed}"
        self._k = Kokoro(c.get("model", "models/kokoro-v1.0.int8.onnx"), c.get("voices", "models/voices-v1.0.bin"))

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._k.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        return np.asarray(samples, dtype=np.float32), int(sr)
