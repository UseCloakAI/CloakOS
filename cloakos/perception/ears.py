"""Ears: loud-sound detection from the PC's microphone (optional).

The SDK doesn't expose Vector's mic stream, so the PC listens instead. Keep the mic
near the robot. Needs `pip install .[ears]`; disabled unless senses.ears.enabled.
"""
from __future__ import annotations

import logging

import numpy as np

from ..bus import Bus

log = logging.getLogger(__name__)


class Ears:
    def __init__(self, cfg: dict, bus: Bus) -> None:
        c = (cfg.get("senses", {}) or {}).get("ears", {}) or {}
        self.enabled = bool(c.get("enabled", False))
        self.threshold_db = float(c.get("threshold_db", -18))
        self.device = c.get("device")
        self.bus = bus
        self._stream = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import sounddevice as sd
        except ImportError:
            log.warning("ears enabled but sounddevice isn't installed (pip install .[ears])")
            return

        def cb(indata, _frames, _time, _status):
            rms = float(np.sqrt(np.mean(np.square(indata, dtype=np.float32))))
            db = 20 * np.log10(max(rms, 1e-9))
            if db > self.threshold_db:
                self.bus.emit_threadsafe("raw.loud_sound", level=round(db, 1))

        self._stream = sd.InputStream(channels=1, samplerate=16000, blocksize=1600, device=self.device, callback=cb)
        self._stream.start()
        log.info("ears listening (threshold %.0f dBFS)", self.threshold_db)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
