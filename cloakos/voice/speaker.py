"""Voice: trigger -> line -> TTS -> 16 kHz PCM -> Vector's speaker (streamed by our own client).

Latency plan:
  - every fallback line is pre-rendered to the disk cache at startup, so those play instantly
  - LLM lines race a timeout; if the LLM is slow we fall back to a cached line
  - every rendered line is cached by text, so repeats are free
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
import wave
from pathlib import Path
from typing import Any

from ..bus import Bus, Event
from ..connection import Body
from .audio import to_vector_wav
from .llm import LineWriter
from .tts import make_tts

log = logging.getLogger(__name__)


def louder(pcm: bytes) -> bytes:
    """TTS comes out quiet for Vector's tiny speaker: bring speech up to full scale with a gentle
    compressor (quiet syllables lifted more than loud ones) so it's clear without crackling."""
    import numpy as np

    x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
    if not len(x):
        return pcm
    peak = float(np.percentile(np.abs(x), 99.7)) or 1e-6      # ignore the odd click when normalising
    x = np.clip(x / peak, -1.5, 1.5)
    x = np.tanh(1.6 * x) / np.tanh(1.6)
    return (x * 0.97 * 32767).astype("<i2").tobytes()


class Voice:
    def __init__(self, cfg: dict, bus: Bus, body: Body, drives: dict[str, float]) -> None:
        c = cfg.get("voice", {}) or {}
        self.enabled = bool(c.get("enabled", True))
        self.volume = int(c.get("volume", 100))
        self.pitch = float(c.get("pitch", 1.15))
        self.cooldown_s = float(c.get("cooldown_s", 8))
        self.llm_ratio = float(c.get("llm_ratio", 0.6))
        self.cache_dir = Path(c.get("cache_dir", "data/voice_cache"))
        self.cache_max = int(c.get("cache_max_files", 500))
        self.triggers: dict[str, dict[str, Any]] = c.get("triggers", {}) or {}
        self.bus, self.body, self.drives = bus, body, drives
        self.llm = LineWriter(cfg)
        self._cfg = cfg
        self.tts = None
        self.tts_error: str | None = None
        self._last_any = 0.0
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()
        bus.on("drive.*", self._on_event)
        bus.on("sense.*", self._on_event)

    @property
    def speaking(self) -> bool:
        return self._lock.locked()

    async def start(self) -> None:
        """Load TTS and pre-render every fallback line (in a worker thread)."""
        if not self.enabled:
            return
        try:
            self.tts = await asyncio.to_thread(make_tts, self._cfg)
        except Exception as e:
            self.tts_error = f"{type(e).__name__}: {e}"
            log.warning("TTS unavailable (%s); voice disabled until fixed", self.tts_error)
            return
        lines = {ln for t in self.triggers.values() for ln in t.get("lines", [])}
        for ln in lines:
            await asyncio.to_thread(self._render, ln)
        log.info("voice ready: %s, %d lines cached", self.tts.id, len(lines))

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha1(f"{self.tts.id}|{self.pitch}|{text.lower()}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.wav"

    def _render(self, text: str) -> Path:
        path = self._cache_path(text)
        if path.exists():
            path.touch()
            return path
        samples, sr = self.tts.synth(text)
        to_vector_wav(samples, sr, path, pitch=self.pitch)
        self._prune()
        return path

    def _prune(self) -> None:
        files = sorted(self.cache_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - self.cache_max)]:
            p.unlink(missing_ok=True)

    # ── triggers ───────────────────────────────────────────────

    def _on_event(self, ev: Event) -> None:
        key = ev.topic.split(".", 1)[1]          # "drive.fear.high" -> "fear.high", "sense.cliff" -> "cliff"
        trig = self.triggers.get(key)
        if not trig or not self.enabled or self.tts is None:
            return
        now = time.time()
        if now - self._last_any < self.cooldown_s or now - self._last.get(key, 0) < float(trig.get("cooldown_s", 20)):
            return
        if random.random() > float(trig.get("chance", 1.0)):
            return
        self._last_any = self._last[key] = now
        asyncio.ensure_future(self._speak_for(key, trig))

    async def _speak_for(self, key: str, trig: dict[str, Any]) -> None:
        pool = trig.get("lines") or []
        text, source = None, "cache"
        if self.llm.enabled and (not pool or random.random() < self.llm_ratio):
            text = await self.llm.line(trig.get("hint", key), dict(self.drives))
            source = "llm"
        if not text:
            if not pool:
                return
            text, source = random.choice(pool), "cache"
        await self.say(text, trigger=key, source=source)

    async def say(self, text: str, trigger: str = "manual", source: str = "manual") -> bool:
        if self.tts is None or not text.strip():
            return False
        if self._lock.locked():
            return False   # never queue speech; a stale line is worse than none
        async with self._lock:
            t0 = time.time()
            path = await asyncio.to_thread(self._render, text)
            ready_ms = int((time.time() - t0) * 1000)
            self.bus.emit("voice.said", text=text, trigger=trigger, source=source, ready_ms=ready_ms, spoken=self.body.ready)
            with wave.open(str(path)) as w:
                pcm, rate = w.readframes(w.getnframes()), w.getframerate()
            await self.body.play_pcm(louder(pcm), self.volume, text=text, rate=rate)
        return True

    async def reply(self, heard: str) -> str | None:
        """Talk-to-it: the human types, the creature answers in character."""
        text = await self.llm.reply(heard, dict(self.drives))
        if text:
            await self.say(text, trigger="reply", source="llm")
        return text

    async def aclose(self) -> None:
        await self.llm.aclose()
