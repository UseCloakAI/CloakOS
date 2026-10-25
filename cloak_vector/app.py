"""Brain: wires everything together and runs the 10 Hz loop.

Modes:
  auto    we hold behavior control, RulePolicy drives the body
  teleop  we hold control, you drive from the web UI
  idle    control released, stock Vector runs; drives still tick and the logger still works
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any

from .behaviors import Obs, Policy, RulePolicy
from .bus import Bus, Event
from .connection import Body, RobotLink
from .datalog import Recorder
from .drives import Context, DriveModel
from .perception import Ears, Senses
from .teleop import TeleopPolicy
from .voice import Voice
from .world import World

log = logging.getLogger(__name__)

MODES = ("auto", "teleop", "idle")
# sense events that count as "something happened" for boredom
NON_STIMULI = {"face_lost", "put_down"}


class Brain:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.hz = float(cfg.get("tick_hz", 10))
        self.bus = Bus()
        self.world = World()
        self.link = RobotLink(cfg, self.bus, self.world)
        self.body = Body(self.link)
        self.drives = DriveModel(cfg)
        self.senses = Senses(cfg, self.bus, self.world)
        self.ears = Ears(cfg, self.bus)
        self.voice = Voice(cfg, self.bus, self.body, self.drives.values)
        self.recorder = Recorder(cfg)
        self.policies: dict[str, Policy] = {
            "auto": RulePolicy(cfg, self.body, self.bus, self.world, self.drives.values),
            "teleop": TeleopPolicy(cfg, self.body),
        }
        self.mode = "idle"
        self.policy: Policy | None = None
        self._auto_rec = False
        self._tick_events: list[dict[str, Any]] = []
        self._stop = asyncio.Event()
        self.bus.on("sense.*", self._on_sense)
        self.bus.on("*", self._collect)

    @property
    def teleop(self) -> TeleopPolicy:
        return self.policies["teleop"]  # type: ignore[return-value]

    # ── events ─────────────────────────────────────────────────

    def _on_sense(self, ev: Event) -> None:
        name = ev.topic.split(".", 1)[1]
        if name not in NON_STIMULI:
            self.world.stimulate()
        self._emit_crossings(self.drives.impulse(name))

    def _emit_crossings(self, crossings: list[str]) -> None:
        for c in crossings:
            self.bus.emit(f"drive.{c}", value=self.drives[c.split(".")[0]])

    def _collect(self, ev: Event) -> None:
        if not ev.topic.startswith("raw."):
            self._tick_events.append({"topic": ev.topic, "t": ev.t, **{k: v for k, v in ev.data.items() if k != "scores"}})

    # ── modes ──────────────────────────────────────────────────

    async def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        if self.policy is not None:
            await self.policy.stop()
        self.policy = self.policies.get(mode)
        self.mode = mode
        await self.link.set_control(mode != "idle")
        if self.policy is not None:
            await self.policy.start()
        auto_rec = (self.cfg.get("datalog", {}) or {}).get("auto_record", [])
        if self._auto_rec and self.recorder.active:
            self._stop_recording()
        self._auto_rec = False
        if mode in auto_rec and not self.recorder.active:
            self.start_recording()
            self._auto_rec = True
        self.bus.emit("mode.changed", mode=mode)

    def start_recording(self, note: str = "") -> None:
        d = self.recorder.start(self.mode, self.hz, note=note)
        self.bus.emit("record.started", dir=str(d))

    def _stop_recording(self) -> None:
        steps = self.recorder.steps
        d = self.recorder.stop()
        self.bus.emit("record.stopped", dir=str(d), steps=steps)

    def toggle_recording(self, on: bool, note: str = "") -> None:
        if on and not self.recorder.active:
            self.start_recording(note)
        elif not on and self.recorder.active:
            self._stop_recording()
        self._auto_rec = False

    # ── loop ───────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        period = 1.0 / self.hz
        last = time.monotonic()
        while not self._stop.is_set():
            start = time.monotonic()
            dt, last = start - last, start
            self.link.poll()
            self.senses.update()
            w = self.world
            ctx = Context(
                moving=w.wheels_moving or abs(self.body.cmd.left_mmps) + abs(self.body.cmd.right_mmps) > 1,
                resting=bool(self.policy and self.policy.resting),
                idle_s=w.idle_s(),
                battery_low=w.battery_low,
            )
            self._emit_crossings(self.drives.update(dt, ctx))
            obs = Obs(time.time(), self.drives.values, w)
            if self.policy is not None and self.link.has_control:
                try:
                    await self.policy.step(obs)
                except Exception:
                    log.exception("policy step failed")
            self._record(obs)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - start)))

    def _record(self, obs: Obs) -> None:
        events, self._tick_events = self._tick_events, []
        discrete = self.body.drain_discrete()
        if not self.recorder.active:
            return
        self.recorder.record(
            {
                "t": round(obs.t, 3),
                "mode": self.mode,
                "obs": self.world.sensors(),
                "drives": self.drives.snapshot(),
                "behavior": self.policy.current if self.policy else "-",
                "action": self.body.cmd.to_dict(),
                "action_source": self.body.source,
                "events": events + discrete,
            },
            frame=self.world.frame,
            frame_id=self.world.frame_id,
        )

    def state(self) -> dict[str, Any]:
        p = self.policy
        return {
            "t": time.time(),
            "connection": self.link.state,
            "has_control": self.link.has_control,
            "mode": self.mode,
            "behavior": p.current if p else "-",
            "scores": getattr(p, "scores", {}),
            "drives": self.drives.snapshot(),
            "sensors": self.world.sensors(),
            "action": self.body.cmd.to_dict(),
            "recording": {"on": self.recorder.active, "steps": self.recorder.steps,
                          "dir": str(self.recorder.dir) if self.recorder.dir else None},
            "voice": {"ready": self.voice.tts is not None, "speaking": self.voice.speaking,
                      "error": self.voice.tts_error, "llm": self.voice.llm.enabled},
        }

    # ── run ────────────────────────────────────────────────────

    async def run(self, mode: str = "auto", web: bool = True) -> None:
        loop = asyncio.get_running_loop()
        self.bus.bind_loop(loop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:   # Windows
                pass
        self.ears.start()
        tasks = [
            asyncio.create_task(self.link.run(), name="link"),
            asyncio.create_task(self._tick_loop(), name="tick"),
            asyncio.create_task(self.voice.start(), name="voice"),
        ]
        if web:
            from .web.server import serve
            tasks.append(asyncio.create_task(serve(self), name="web"))
        await self.set_mode(mode)
        try:
            await self._stop.wait()
        finally:
            await self.shutdown(tasks)

    def request_stop(self) -> None:
        self._stop.set()

    async def shutdown(self, tasks: list[asyncio.Task]) -> None:
        log.info("shutting down")
        self._stop.set()
        if self.policy is not None:
            try:
                await asyncio.wait_for(self.policy.stop(), 5)
            except Exception:
                pass
        self.recorder.stop()
        self.ears.stop()
        await self.link.stop()
        link_task = next((t for t in tasks if t.get_name() == "link"), None)
        for t in tasks:
            if t is not link_task:
                t.cancel()
        if link_task is not None:
            try:
                await asyncio.wait_for(link_task, 15)   # lets it stop motors + disconnect cleanly
            except Exception:
                link_task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.voice.aclose()
