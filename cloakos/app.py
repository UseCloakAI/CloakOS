"""Brain: wires everything together and runs the 10 Hz loop.

Modes:
  auto    we hold behavior control, RulePolicy drives the body
  teleop  we hold control, you drive from the web UI
  idle    control released, stock Vector runs; drives still tick and the logger still works
"""
from __future__ import annotations

import asyncio
import os
import logging
import signal
import time
from typing import Any

from .behaviors import Obs, Policy, RulePolicy
from .bus import Bus, Event
from .connection import Body, RobotLink
from .datalog import Recorder
from .drives import Context, DriveModel
from .expression import Emotes, Eyes
from .perception import Ears, Senses
from .teleop import TeleopPolicy
from .vector import rgb565
from .voice import Voice
from .world import World

log = logging.getLogger(__name__)

MODES = ("auto", "teleop", "idle", "learned", "goto", "home")
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
        ex = cfg.get("expression", {}) or {}
        self.eyes = Eyes(cfg)
        self.eyes_on = bool(ex.get("eyes", True))
        self.eye_fps = min(12.0, float(ex.get("eye_fps", 10)))     # more crashes Vector's display process
        self.eye_frame = None                    # latest rendered eyes (PIL), for the dashboard too
        self.reactions: dict[str, str] = ex.get("reactions", {}) or {}
        self.body.emotes = Emotes(cfg, self.eyes)
        from .expression.sfx import Sfx
        self.sfx = Sfx(cfg, self.bus, self.body, self.drives)
        self._emote_task: asyncio.Task | None = None
        self.recorder = Recorder(cfg)
        self.policies: dict[str, Policy] = {
            "auto": RulePolicy(cfg, self.body, self.bus, self.world, self.drives.values),
            "teleop": TeleopPolicy(cfg, self.body),
        }
        if not (cfg.get("share", {}) or {}).get("enabled", True):
            os.environ["CLOAK_SHARE"] = "0"            # training subprocesses read this too
        from .rl.manager import Trainer
        from .rl.policy import LearnedPolicy
        self.policies["learned"] = LearnedPolicy(self.body, self.link)
        self.trainer = Trainer((cfg.get("datalog", {}) or {}).get("dir", "data"))
        from .mapping.mapper import Mapper
        self.mapper = Mapper(cfg, self.link, self.world, self.bus)
        self.mapper.vision_needed = lambda: self.mode in ("auto", "goto", "learned") or bool(self.tester and self.tester.running)
        from .behaviors.goto import GoToPolicy
        from .rl.skills import Skills
        self.skills = Skills(self.link, self.world, self.mapper)
        self.body.skills = self.skills
        self.policies["goto"] = GoToPolicy(self.body, self.skills, self.bus)
        self._goto_return = "auto"
        self.stuck = False                       # froze itself because it was stuck
        self._stuck_s = 0.0
        self._stuck_ping = 0.0
        self.bus.on("goto.*", self._on_goto)
        self.mode = "idle"
        self.policy: Policy | None = None
        self._auto_rec = False
        self.restart = False
        self._tick_events: list[dict[str, Any]] = []
        self._stop = asyncio.Event()
        self.frozen = False
        from .calibrate import Calibrator
        self.calibrator = Calibrator(self)
        self.tester = None                       # sim.tester.Tester, made on first use
        fz = cfg.get("freeze", {}) or {}
        self.freeze_taps = int(fz.get("taps", 3))
        self.freeze_window = float(fz.get("window_s", 1.5))
        self._taps: list[float] = []
        self.bus.on("raw.button_press", self._on_button)
        self.bus.on("raw.edge", lambda ev: self.eyes.alert("edge", 1.6))
        self.bus.on("sense.*", self._on_sense)
        self.bus.on("*", self._collect)

    @property
    def learned(self):
        return self.policies["learned"]

    @property
    def teleop(self) -> TeleopPolicy:
        return self.policies["teleop"]  # type: ignore[return-value]

    # ── events ─────────────────────────────────────────────────

    def _on_sense(self, ev: Event) -> None:
        name = ev.topic.split(".", 1)[1]
        if name not in NON_STIMULI:
            self.world.stimulate()
        self._emit_crossings(self.drives.impulse(name))
        if name in self.reactions and self.mode == "auto":
            self.react(self.reactions[name])

    def _on_button(self, ev: Event) -> None:
        """Triple tap on the back button toggles freeze."""
        self._taps = [t for t in self._taps if ev.t - t < self.freeze_window] + [ev.t]
        if len(self._taps) >= self.freeze_taps:
            self._taps = []
            asyncio.ensure_future(self.set_frozen(not self.frozen, source="button"))

    async def set_frozen(self, on: bool, source: str = "ui") -> None:
        """Freeze: every motor stops and stays stopped (behaviors, teleop, emotes all blocked).
        Control is kept, so stock Vector doesn't wander off either. Eyes go still."""
        if on == self.frozen:
            return
        self.frozen = self.body.frozen = self.eyes.frozen = on
        if not on:
            self.stuck = False
        if on:
            if self.policy is not None:
                await self.policy.stop()
            await self.body.stop()
        elif self.policy is not None:
            await self.policy.start()
        log.info("freeze %s (%s)", "on" if on else "off", source)
        self.bus.emit("freeze.changed", on=on, source=source)

    # ── talking + going places ─────────────────────────────────

    async def command(self, text: str) -> dict:
        """You said something: the LLM (or simple rules) decides what to say and do."""
        from .voice.commander import decide

        here = tuple(self.policies["goto"].skills.snapshot()["pos"][:2]) if self.link.connected else None
        d = await decide(self.voice.llm, text, self.mapper.known(), here)
        if d.action == "go_to" and d.target:
            ok = await self.go_to(d.target)
            if not ok:
                d.say = d.say or "I can't get there right now."
        elif d.action == "stop":
            await self.set_frozen(True, source="voice")
        elif d.action == "go_home":
            asyncio.ensure_future(self.go_home())
        elif d.action == "moves" and d.steps:
            asyncio.ensure_future(self.run_moves(d.steps))
        elif d.action == "wander":
            if self.frozen:
                await self.set_frozen(False, source="voice")
            await self.set_mode("auto")
        if d.say:
            asyncio.ensure_future(self.voice.say(d.say, trigger="reply", source=d.source))
        self.bus.emit("command", text=text, say=d.say, action=d.action, target=d.target, source=d.source)
        return {"say": d.say, "action": d.action, "target": d.target, "source": d.source}

    MOVE_WORDS = {"forward": "Forward", "back": "Backing up", "turn": "Turning", "head": "Looking", "lift": "Lift", "wait": "Waiting"}

    async def run_moves(self, steps: list) -> None:
        """Precise steps from a command or the LLM, one after another. Autonomy pauses meanwhile (teleop
        holds control without driving); the edge stop still guards every forward step."""
        if not self.link.connected:
            return
        if self.frozen:
            await self.set_frozen(False, source="moves")
        back_to = self.mode if self.mode in ("auto", "idle", "teleop") else "auto"
        await self.set_mode("teleop")
        try:
            for kind, val in steps:
                if self.frozen:                          # "stop" (or stuck) cancels the rest
                    break
                self.eyes.flash(self.MOVE_WORDS.get(kind, kind), 1.2)
                if kind == "forward":
                    await self.body.straight(val, 80)
                elif kind == "back":
                    await self.body.straight(-val, 80)
                elif kind == "turn":
                    await self.body.turn(val, 90)
                elif kind == "head":
                    await self.body.head_angle(val, 8)
                elif kind == "lift":
                    await self.body.lift_height(val, 8)
                elif kind == "wait":
                    await asyncio.sleep(val)
        finally:
            await self.set_mode(back_to)

    async def go_to(self, object_id: str) -> bool:
        obj = next((o for o in self.mapper.objects if o["id"] == object_id), None) or self.mapper.find(object_id)
        if obj is None or not self.link.connected:
            return False
        if self.frozen:
            await self.set_frozen(False, source="go_to")
        gp = self.policies["goto"]
        gp.set_target(obj, tuple(self.skills.snapshot()["pos"][:2]))
        if self.mode != "goto":
            self._goto_return = self.mode if self.mode in ("auto", "idle") else "auto"
            await self.set_mode("goto")
        else:
            await gp.start()
        return True

    def _on_goto(self, ev: Event) -> None:
        how = ev.topic.split(".", 1)[1]
        if how in ("arrived", "failed"):
            self.react("happy" if how == "arrived" else "sad")
            self.eyes.flash("Got there!" if how == "arrived" else "Couldn't reach it", 2.0)
            asyncio.get_running_loop().call_later(2.0, lambda: asyncio.ensure_future(self.set_mode(self._goto_return)))

    def react(self, emote: str) -> None:
        """Fire-and-forget emote on top of whatever's running; skipped if one is already playing."""
        if self._emote_task is None or self._emote_task.done():
            self._emote_task = asyncio.ensure_future(self.body.emote(emote))

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
            self._check_stuck(dt)
            if w.battery_low and self.mode == "auto" and not w.on_charger and not self.frozen:
                asyncio.ensure_future(self.go_home())          # tired: go charge (Vector's own dock, our eyes)
            obs = Obs(time.time(), self.drives.values, w)
            if self.policy is not None and self.link.has_control and not self.frozen:
                try:
                    await self.policy.step(obs)
                except Exception:
                    log.exception("policy step failed")
            self._record(obs)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - start)))

    async def _eye_loop(self) -> None:
        """Render Cloak's eyes every frame; push them to the screen whenever we hold control."""
        # Vector's display process (vic-anim) crashes (error 8xx) if it's flooded: send at most
        # eye_fps, skip frames that didn't change, and slow down when Vector is slow to take them.
        period = 1.0 / self.eye_fps
        sent, last_sent, slow = None, 0.0, 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            face = self.world.face
            gaze = (face.cx, face.cy) if face is not None and self.world.face_visible(1.0) else None
            behavior = self.policy.current if self.policy else "-"
            self.eyes.status = self.activity()
            self.eye_frame = self.eyes.frame(self.eyes.shape(self.drives.values, behavior, gaze))
            if self.eyes_on and self.body.ready:
                px = rgb565(self.eye_frame)
                if px != sent or t0 - last_sent > 1.0:          # refresh now and then so it never times out
                    await self.body.face(px, 2500)
                    sent, last_sent = px, t0
                    took = time.monotonic() - t0
                    slow = slow + 1 if took > period * 0.8 else max(0, slow - 1)
                    if slow > 5:                                 # Vector is struggling: back off
                        period, slow = min(0.25, period * 1.3), 0
                        log.info("eyes: Vector is slow to draw, now %.0f fps", 1 / period)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _check_stuck(self, dt: float) -> None:
        """Wheels told to move but nothing happens: pushing against something, or a tread caught.
        Two signs: the wheels themselves don't turn, or it's told to turn and the gyro says it isn't.
        Then everything stops (freeze) and it pings for help until someone unfreezes it."""
        w, cmd = self.world, self.body.cmd
        if self.frozen:
            if self.stuck and time.time() - self._stuck_ping > 3.0:     # keep calling for help
                self._stuck_ping = time.time()
                self.bus.emit("stuck")
                self.eyes.alert("stuck", 2.5)
            self._stuck_s = 0.0
            return
        want = (abs(cmd.left_mmps) + abs(cmd.right_mmps)) / 2
        got = (abs(w.left_mmps or 0.0) + abs(w.right_mmps or 0.0)) / 2
        want_turn = abs(cmd.right_mmps - cmd.left_mmps) > 60 and abs(cmd.right_mmps + cmd.left_mmps) < 40
        turning = abs((w.gyro or (0, 0, 0))[2]) > 0.08
        stalled = want > 40 and got < 0.25 * want
        spun_out = want_turn and not turning
        moving_mode = (self.mode in ("auto", "goto", "learned") and self.link.has_control and not w.picked_up
                       and not w.cliff and time.time() - w.edge_t > 1.5)        # the edge stop holds wheels on purpose
        self._stuck_s = self._stuck_s + dt if (moving_mode and (stalled or spun_out)) else 0.0
        if self._stuck_s > 1.8:
            self._stuck_s = 0.0
            self.stuck = True
            self._stuck_ping = time.time()
            log.warning("stuck (%s): stopping everything", "wheels stalled" if stalled else "not turning")
            asyncio.ensure_future(self.set_frozen(True, source="stuck"))
            self.bus.emit("stuck")
            self.eyes.alert("stuck", 2.5)

    async def go_home(self) -> bool:
        """Back to the charger with Vector's own docking, while CloakOS keeps the eyes and holds control.
        Stays in 'home' mode (resting) afterwards; switch to Auto to head out again."""
        if self.mode == "home" or not self.link.connected:
            return False
        if self.frozen:
            await self.set_frozen(False, source="go_home")
        await self.set_mode("home")
        self._homing = True
        try:
            ok = await self.body.go_home()
        finally:
            self._homing = False
        self.eyes.flash("Home!" if ok or self.world.on_charger else "Couldn't find home", 2.0)
        self.react("happy" if ok or self.world.on_charger else "sad")
        return ok

    ACTIVITY = {"wander": "Wandering", "look_around": "Looking around", "approach_face": "Coming to say hi"}

    def activity(self) -> str | None:
        """What it's doing, in a few words, for the screen. None = just eyes."""
        w = self.world
        if self.frozen:
            return "Stuck!" if self.stuck else "Frozen"
        if self.mode == "home":
            return "Going home" if getattr(self, "_homing", False) else ("Charging" if w.charging else None)
        if w.on_charger and (abs(w.left_mmps or 0) > 5 or abs(w.right_mmps or 0) > 5):
            return "Undocking"
        if self.calibrator.running:
            return "Calibrating"
        if self.tester is not None and self.tester.running:
            return "Running a test"
        if self.mode == "goto":
            gp = self.policies["goto"]
            return f"Going to the {gp.target['label']}" if gp.target and gp.status == "driving" else None
        if self.mode == "teleop":
            return "Remote control" if self.body.cmd.left or self.body.cmd.right else None
        if self.mode in ("auto", "learned") and self.policy is not None:
            return self.ACTIVITY.get(self.policy.current)
        return None

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
            "link_hint": self.link.hint,
            "frozen": self.frozen,
            "target": self.link.target,
            "volume": self.link.volume,
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
            asyncio.create_task(self._eye_loop(), name="eyes"),
            asyncio.create_task(self.voice.start(), name="voice"),
            asyncio.create_task(self.mapper.run(), name="mapper"),
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

    def request_restart(self) -> None:
        """Stop cleanly, then __main__ re-execs the process (used after the setup wizard saves)."""
        self.restart = True
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
        await asyncio.to_thread(self.mapper.save)
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
