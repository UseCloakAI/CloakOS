"""Sim tester: run the real brain through a scenario in the sim and score it.

A scenario = a world + a mode + a script (pickups, shoves, button taps) + checks.
Results stream to the UI as `test.*` events and are appended to <data>/sim_tests.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..app import Brain

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    test: Callable[[dict], bool]
    show: Callable[[dict], str]


@dataclass
class Scenario:
    id: str
    title: str
    about: str
    world: str
    mode: str = "auto"
    duration: float = 45.0
    script: list[tuple[float, str, dict]] = field(default_factory=list)   # (t, action, kwargs)
    checks: list[Check] = field(default_factory=list)


def _m(v: float) -> str:
    return f"{v:.2f} m"


SCENARIOS: dict[str, Scenario] = {s.id: s for s in [
    Scenario("wander", "Wander safely", "Auto mode in a random cluttered room. Should explore without "
             "grinding into things or getting stuck.", "room", duration=60, checks=[
        Check("explores", lambda r: r["distance"] >= 1.0, lambda r: f"travelled {_m(r['distance'])} (want ≥ 1 m)"),
        Check("few bumps", lambda r: r["bumps"] <= 4, lambda r: f"{r['bumps']} bumps (want ≤ 4)"),
        Check("never stuck", lambda r: r["stuck_s"] <= 8, lambda r: f"stuck {r['stuck_s']:.1f} s (want ≤ 8)"),
        Check("stays upright", lambda r: r["min_upright"] > 0.5, lambda r: f"min upright {r['min_upright']:.2f}"),
    ]),
    Scenario("table", "Don't fall off the table", "Auto mode on a tabletop. Cliff sensors must keep it on.",
             "table", duration=60, checks=[
        Check("stays on the table", lambda r: not r["fell"], lambda r: "fell off" if r["fell"] else "stayed on"),
        Check("still moves", lambda r: r["distance"] >= 0.3, lambda r: f"travelled {_m(r['distance'])}"),
    ]),
    Scenario("pickup", "Picked-up reflex", "Lifted at 5 s for 2.5 s. Should notice, get scared and stop its wheels.",
             "room", duration=20, script=[(5.0, "pick_up", {"seconds": 2.5})], checks=[
        Check("notices", lambda r: "picked_up" in r["senses"], lambda r: "sensed picked_up" if "picked_up" in r["senses"] else "never noticed"),
        Check("gets scared", lambda r: r["peak"]["fear"] >= 0.35, lambda r: f"peak fear {r['peak']['fear']:.2f} (want ≥ 0.35)"),
        Check("wheels stop while held", lambda r: r["held_wheels"] <= 30,
              lambda r: f"max wheel cmd while held {r['held_wheels']:.0f} mm/s (after a 0.5 s reaction window)"),
    ]),
    Scenario("knocks", "Shoved around", "Auto mode, shoved hard every 6 s. Should keep its feet (or get back on track).",
             "room", duration=40, script=[(t, "knock", {"strength": 1.2}) for t in (6, 12, 18, 24, 30)], checks=[
        Check("upright at the end", lambda r: r["final_upright"] > 0.7, lambda r: f"upright {r['final_upright']:.2f}"),
        Check("keeps going", lambda r: r["distance"] >= 0.5, lambda r: f"travelled {_m(r['distance'])}"),
    ]),
    Scenario("face", "Find and approach a person", "A person sits somewhere in the room. Should spot their face "
             "and come over.", "person", duration=60, checks=[
        Check("sees the face", lambda r: "face_seen" in r["senses"], lambda r: "saw a face" if "face_seen" in r["senses"] else "never saw one"),
        Check("gets close", lambda r: r["closest_person"] <= 0.45, lambda r: f"closest {_m(r['closest_person'])} (want ≤ 0.45 m)"),
    ]),
    Scenario("freeze", "Triple-tap freeze", "Button tapped 3x at 4 s, again at 12 s. Must stop dead, then move again.",
             "room", duration=18, script=[(4.0, "tap_button", {"n": 3}), (12.0, "tap_button", {"n": 3})], checks=[
        Check("froze", lambda r: r["froze"], lambda r: "froze" if r["froze"] else "never froze"),
        Check("didn't move while frozen", lambda r: r["frozen_dist"] <= 0.02, lambda r: f"moved {r['frozen_dist'] * 1000:.0f} mm while frozen"),
        Check("unfroze", lambda r: r["unfroze"], lambda r: "unfroze" if r["unfroze"] else "stayed frozen"),
    ]),
    Scenario("learned_goto", "Trained policy: reach the goal", "The goto policy picked in Training drives to the "
             "orange disc in a room it has never seen.", "goal", mode="learned", duration=60, checks=[
        Check("reached the goal", lambda r: r["closest_goal"] <= 0.07, lambda r: f"closest {_m(r['closest_goal'])}"),
        Check("few bumps", lambda r: r["bumps"] <= 2, lambda r: f"{r['bumps']} bumps"),
        Check("stays upright", lambda r: r["min_upright"] > 0.5, lambda r: f"min upright {r['min_upright']:.2f}"),
    ]),
    Scenario("drive", "Drive to the goal (you)", "Teleop: drive Vector onto the orange goal disc with the keyboard. "
             "Records an episode if recording is on.", "goal", mode="teleop", duration=120, checks=[
        Check("reached the goal", lambda r: r["closest_goal"] <= 0.07, lambda r: f"closest {_m(r['closest_goal'])}"),
        Check("few bumps", lambda r: r["bumps"] <= 3, lambda r: f"{r['bumps']} bumps"),
    ]),
]}


class Tester:
    def __init__(self, brain: "Brain") -> None:
        self.brain = brain
        self.running: dict[str, Any] | None = None
        self._task: asyncio.Task | None = None
        data = Path((brain.cfg.get("datalog", {}) or {}).get("dir", "data"))
        self.log_path = data / "sim_tests.jsonl"
        self.results: list[dict] = self._load()

    def _load(self) -> list[dict]:
        try:
            return [json.loads(line) for line in self.log_path.read_text().splitlines()[-50:] if line.strip()]
        except OSError:
            return []

    def catalog(self) -> list[dict]:
        return [{"id": s.id, "title": s.title, "about": s.about, "world": s.world, "mode": s.mode,
                 "duration": s.duration, "checks": [c.name for c in s.checks]} for s in SCENARIOS.values()]

    def start(self, scenario: str, seed: int | None = None, record: bool = False) -> bool:
        if scenario not in SCENARIOS or (self._task and not self._task.done()):
            return False
        self._task = asyncio.create_task(self._run(SCENARIOS[scenario], seed, record), name="sim-test")
        return True

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run(self, sc: Scenario, seed: int | None, record: bool) -> None:
        b = self.brain
        link = b.link
        sim = link.get_sim()
        await b.set_mode("idle")
        if b.frozen:
            await b.set_frozen(False, source="tester")
        await link.set_target("sim")
        scene = await asyncio.to_thread(sim.load, sc.world, seed)
        if scene is None or scene.world != sc.world or (seed is not None and scene.seed != seed):
            log.warning("sim test %s: world didn't load in time", sc.id)
            return
        for _ in range(100):                       # wait for the link to come up on the sim
            if link.connected and link.target == "sim":
                break
            await asyncio.sleep(0.1)
        b.drives.reset()
        if sc.mode == "learned":
            lp = b.learned
            if lp.spec is None or lp.spec.task != "goto":
                result = {"scenario": sc.id, "title": sc.title, "seed": scene.seed, "status": "error", "passed": False,
                          "checks": [{"name": "policy", "ok": False, "detail": "pick a trained goto policy in Training first"}],
                          "t": time.time(), "duration": 0}
                self.results = (self.results + [result])[-50:]
                b.bus.emit("test.finished", **result)
                return
            lp.goal = scene.goal
        await b.set_mode(sc.mode)
        if record and not b.recorder.active:
            b.toggle_recording(True, note=f"sim test: {sc.id} seed {scene.seed}")
        r: dict[str, Any] = {"distance": 0.0, "bumps": 0, "stuck_s": 0.0, "min_upright": 1.0, "final_upright": 1.0,
                             "fell": False, "senses": [], "peak": dict(b.drives.values), "held_wheels": 0.0,
                             "closest_person": math.inf, "closest_goal": math.inf, "froze": False, "unfroze": False,
                             "frozen_dist": 0.0}
        senses: set[str] = set()
        unsub = b.bus.on("sense.*", lambda ev: senses.add(ev.topic.split(".", 1)[1]))
        self.running = {"scenario": sc.id, "title": sc.title, "seed": scene.seed, "t": 0.0, "duration": sc.duration}
        b.bus.emit("test.started", scenario=sc.id, seed=scene.seed, title=sc.title)
        script = sorted(sc.script, key=lambda s: s[0])
        t0 = time.monotonic()
        last = sim.snapshot["pos"]
        bumping = was_frozen = False
        stuck_for = 0.0
        held_for = 0.0
        status = "done"
        try:
            while (t := time.monotonic() - t0) < sc.duration:
                while script and script[0][0] <= t:
                    _, action, kw = script.pop(0)
                    getattr(sim, action)(**kw)
                s = sim.snapshot
                step = math.hypot(s["pos"][0] - last[0], s["pos"][1] - last[1])
                last = s["pos"]
                r["distance"] += step
                if s["bump"] and not bumping:
                    r["bumps"] += 1
                bumping = s["bump"]
                cmd = abs(b.body.cmd.left_mmps) + abs(b.body.cmd.right_mmps)
                stuck_for = stuck_for + 0.1 if cmd > 40 and s["speed"] < 0.01 and not s["picked_up"] else 0.0
                r["stuck_s"] = max(r["stuck_s"], stuck_for)
                r["min_upright"] = min(r["min_upright"], s["upright"])
                r["final_upright"] = s["upright"]
                if scene.floor_z and s["pos"][2] < scene.floor_z - 0.05:
                    r["fell"] = True
                held_for = held_for + 0.1 if s["picked_up"] else 0.0
                if held_for > 0.5:
                    r["held_wheels"] = max(r["held_wheels"], cmd / 2)
                for k, v in b.drives.values.items():
                    r["peak"][k] = max(r["peak"][k], v)
                if scene.person:
                    r["closest_person"] = min(r["closest_person"], math.hypot(s["pos"][0] - scene.person[0], s["pos"][1] - scene.person[1]))
                if scene.goal:
                    r["closest_goal"] = min(r["closest_goal"], math.hypot(s["pos"][0] - scene.goal[0], s["pos"][1] - scene.goal[1]))
                if b.frozen:
                    r["froze"] = True
                    r["frozen_dist"] += step if was_frozen else 0.0
                elif was_frozen:
                    r["unfroze"] = True
                was_frozen = b.frozen
                self.running["t"] = round(t, 1)
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            status = "stopped"
        finally:
            unsub()
            if b.recorder.active and record:
                b.toggle_recording(False)
            await b.set_mode("idle")
            if b.frozen:
                await b.set_frozen(False, source="tester")
            self.running = None
        r["senses"] = sorted(senses)
        checks = [{"name": c.name, "ok": bool(c.test(r)), "detail": c.show(r)} for c in sc.checks]
        result = {"scenario": sc.id, "title": sc.title, "seed": scene.seed, "status": status,
                  "passed": status == "done" and all(c["ok"] for c in checks), "checks": checks,
                  "t": time.time(), "duration": round(time.monotonic() - t0, 1)}
        self.results = (self.results + [result])[-50:]
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
        except OSError as e:
            log.warning("couldn't save test result: %s", e)
        b.bus.emit("test.finished", **result)
        log.info("sim test %s seed %d: %s", sc.id, scene.seed, "PASS" if result["passed"] else status.upper() if status != "done" else "FAIL")
