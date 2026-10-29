"""Recorder: one folder per episode, ready for imitation learning.

data/episodes/<UTC stamp>_<mode>/
  meta.json      fps, camera size, action space, mode, config snapshot (secrets stripped)
  steps.jsonl    one line per tick: obs (sensors), drives, behavior, action, events, frame path
  frames/        <step>.jpg, low-res camera frames

Writes happen on a background thread so the 10 Hz loop never blocks on disk.
A phase-2 exporter can turn this into LeRobot/HDF5 without touching the recorder.
"""
from __future__ import annotations

import copy
import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ACTION_SPACE = {
    "left_mmps": "left wheel speed command, mm/s, +-220",
    "right_mmps": "right wheel speed command, mm/s, +-220",
    "head_rate": "head motor command, rad/s, +-4",
    "lift_rate": "lift motor command, rad/s, +-4",
}
SECRET_KEYS = {"api_key", "serial", "guid"}


def _strip_secrets(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: ("***" if k in SECRET_KEYS and v else _strip_secrets(v)) for k, v in d.items()}
    return d


class Recorder:
    def __init__(self, cfg: dict) -> None:
        c = cfg.get("datalog", {}) or {}
        self.root = Path(c.get("dir", "data")) / "episodes"
        self.frame_size = tuple(c.get("frame_size", (320, 180)))
        self.jpeg_quality = int(c.get("jpeg_quality", 80))
        self.frame_every = max(1, int(c.get("frame_every", 1)))   # save a frame every N ticks
        self._cfg = cfg
        self.dir: Path | None = None
        self.steps = 0
        self.frames = 0
        self.started_at = 0.0
        self._q: queue.Queue = queue.Queue(maxsize=600)
        self._thread: threading.Thread | None = None
        self._last_frame_id = -1

    @property
    def active(self) -> bool:
        return self.dir is not None

    def start(self, mode: str, fps: float, note: str = "") -> Path:
        if self.active:
            self.stop()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = self.root / f"{stamp}_{mode}"
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        meta = {
            "format": "cloakos-vector/v1",
            "created": stamp,
            "mode": mode,
            "note": note,
            "fps": fps,
            "frame_size": list(self.frame_size),
            "action_space": ACTION_SPACE,
            "config": _strip_secrets(copy.deepcopy(dict(self._cfg))),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        self.steps = self.frames = 0
        self.started_at = time.time()
        self._last_frame_id = -1
        self._thread = threading.Thread(target=self._writer, args=(self.dir, self._q), daemon=True)
        self._thread.start()
        log.info("recording -> %s", self.dir)
        return self.dir

    def record(self, step: dict[str, Any], frame: Any = None, frame_id: int = -1) -> None:
        if not self.active:
            return
        i = self.steps
        self.steps += 1
        rel = None
        if frame is not None and frame_id != self._last_frame_id and i % self.frame_every == 0:
            self._last_frame_id = frame_id
            rel = f"frames/{i:07d}.jpg"
            self.frames += 1
        try:
            self._q.put_nowait((i, rel, frame if rel else None, step))
        except queue.Full:
            log.warning("recorder queue full, dropping step %d", i)

    def stop(self) -> Path | None:
        d, self.dir = self.dir, None
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=10)
            self._thread = None
        if d:
            log.info("recording saved: %s (%d steps, %d frames)", d, self.steps, self.frames)
        return d

    def _writer(self, d: Path, q: queue.Queue) -> None:
        with open(d / "steps.jsonl", "a", buffering=1) as f:
            while True:
                item = q.get()
                if item is None:
                    break
                i, rel, frame, step = item
                if rel and frame is not None:
                    try:
                        frame.convert("RGB").resize(self.frame_size).save(d / rel, quality=self.jpeg_quality)
                    except Exception as e:
                        log.debug("frame save failed: %s", e)
                        rel = None
                f.write(json.dumps({"i": i, **step, "frame": rel}, default=str) + "\n")

    def delete(self, name: str) -> bool:
        """Delete one saved episode by folder name. Never the one being recorded."""
        import shutil

        d = (self.root / name).resolve()
        if d.parent != self.root.resolve() or not d.is_dir() or (self.active and self.dir and d == self.dir.resolve()):
            return False
        shutil.rmtree(d)
        return True

    def delete_all(self) -> int:
        """Every recording except one being written right now."""
        n = 0
        for e in self.list_episodes(limit=100_000):
            n += self.delete(e["name"])
        return n

    def list_episodes(self, limit: int = 30) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir(), reverse=True)[:limit]:
            steps = d / "steps.jsonl"
            n = sum(1 for _ in open(steps)) if steps.exists() else 0
            out.append({"name": d.name, "steps": n, "frames": len(list((d / "frames").glob("*.jpg"))), "created": d.stat().st_mtime})
        return out
