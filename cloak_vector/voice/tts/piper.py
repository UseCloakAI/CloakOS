"""Piper via piper-tts. pip install .[piper], then download a voice .onnx + .onnx.json (see README)."""
from __future__ import annotations

import numpy as np


class PiperTTS:
    def __init__(self, c: dict) -> None:
        from piper import PiperVoice

        model = c.get("model", "models/en_US-lessac-medium.onnx")
        self.id = f"piper:{model}"
        self._v = PiperVoice.load(model)

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        v = self._v
        if hasattr(v, "synthesize_stream_raw"):   # piper-tts 1.2
            raw = b"".join(v.synthesize_stream_raw(text))
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, int(v.config.sample_rate)
        chunks = list(v.synthesize(text))          # piper-tts 1.3+
        audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(1, np.float32)
        return audio.astype(np.float32), int(chunks[0].sample_rate if chunks else v.config.sample_rate)
