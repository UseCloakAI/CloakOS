import asyncio
import time

from cloakos.behaviors import Obs, RulePolicy
from cloakos.bus import Bus
from cloakos.config import load_config
from cloakos.connection import Action
from cloakos.teleop import TeleopPolicy
from cloakos.world import Face, World


class FakeBody:
    def __init__(self):
        self.cmd = Action()
        self.source = "none"
        self.calls = []

    async def apply(self, a, source=None):
        self.cmd = a.clamped()

    async def drive(self, l, r, source=None):
        self.cmd = Action(l, r)

    async def stop(self):
        self.cmd = Action()

    def __getattr__(self, name):          # anim / head_angle / turn / ... just record
        async def rec(*a, **k):
            self.calls.append(name)
            await asyncio.sleep(0)
        return rec


def drives(**kw):
    d = {"energy": 0.7, "curiosity": 0.4, "boredom": 0.1, "attachment": 0.3, "fear": 0.0}
    d.update(kw)
    return d


async def step(p, d, w):
    await p.step(Obs(time.time(), d, w))
    await asyncio.sleep(0.01)


async def test_picks_face_when_visible_then_startle_interrupts():
    cfg = load_config("config.yaml")
    cfg["behaviors"]["temperature"] = 1e-4          # deterministic pick for the test
    d, w, bus = drives(), World(), Bus()
    seen = []
    bus.on("behavior.changed", lambda ev: seen.append(ev.data["name"]))
    p = RulePolicy(cfg, FakeBody(), bus, w, d)

    await step(p, d, w)
    assert p.current == "wander"

    w.face = Face(id=1, cx=0.5, cy=0.5, size=0.1)
    await step(p, d, w)
    assert p.current == "wander"             # min_s not reached, no interrupt

    p._since -= 100
    await step(p, d, w)
    assert p.current == "approach_face"

    d["fear"] = 0.9                          # reflex cuts in despite min_s
    await step(p, d, w)
    assert p.current == "startle"
    assert seen == ["wander", "approach_face", "startle"]
    await p.stop()


async def test_low_energy_naps_and_marks_resting():
    cfg = load_config("config.yaml")
    d, w = drives(energy=0.1), World()
    p = RulePolicy(cfg, FakeBody(), Bus(), w, d)
    await step(p, d, w)
    assert p.current == "nap" and p.resting
    await p.stop()


async def test_bored_and_lonely_sulks():
    cfg = load_config("config.yaml")
    d, w = drives(boredom=0.95, attachment=0.05, curiosity=0.2, energy=0.5), World()
    p = RulePolicy(cfg, FakeBody(), Bus(), w, d)
    await step(p, d, w)
    assert p.current == "sulk"
    await p.stop()


def test_teleop_mapping_and_deadman():
    cfg = load_config("config.yaml")
    t = TeleopPolicy(cfg, FakeBody())
    t.input(throttle=1, steer=0)
    a = t.action()
    assert a.left_mmps == a.right_mmps == 150
    t.input(throttle=0, steer=1)
    a = t.action()
    assert a.left_mmps > 0 > a.right_mmps
    t._last_input -= 10
    assert t.action() == Action()
