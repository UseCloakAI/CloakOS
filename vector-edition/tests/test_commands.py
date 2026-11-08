"""Talking to Vector: LLM decisions, offline phrasing, and driving to a remembered object."""
import asyncio
import math

import pytest

from cloakos.voice.commander import decide, fallback, parse

OBJS = [{"id": "mug-1", "label": "mug", "pos": [1.0, 0.5, 0.0]}, {"id": "shoe-2", "label": "shoe", "pos": [-1, 0, 0]}]


def test_parse_llm_json():
    d = parse('Sure! {"say": "On my way!", "action": "go_to", "target": "mug-1"}', OBJS)
    assert (d.say, d.action, d.target) == ("On my way!", "go_to", "mug-1")
    d = parse('{"say": "Hmm", "action": "go_to", "target": "the shoe"}', OBJS)      # word instead of id
    assert d.target == "shoe-2"
    d = parse('{"say": "Where?", "action": "go_to", "target": "dragon"}', OBJS)
    assert d.action == "none"
    assert parse("no json here", OBJS) is None


def test_fallback_phrases():
    assert fallback("go to the mug please", OBJS).target == "mug-1"
    assert fallback("Find my shoe!", OBJS).target == "shoe-2"
    assert fallback("go to the dragon", OBJS).action == "none"
    assert fallback("stop", OBJS).action == "stop"
    assert fallback("go explore", OBJS).action == "wander"


class FakeLLM:
    chat_model = "logos"

    def __init__(self, reply):
        self.reply, self.seen = reply, None

    async def ask(self, user, system, **kw):
        self.seen = (user, system, kw)
        return self.reply


async def test_decide_uses_llm_then_rules():
    llm = FakeLLM('{"say": "Mug time!", "action": "go_to", "target": "mug-1"}')
    d = await decide(llm, "can you fetch yourself over by your mug", OBJS, (0.0, 0.0))   # fuzzy: needs the LLM
    assert d.target == "mug-1" and "mug-1: mug" in llm.seen[1] and llm.seen[2]["model"] == "logos"
    d = await decide(FakeLLM(None), "go to the shoe", OBJS, None)          # offline
    assert d.target == "shoe-2" and d.source == "rules"


async def test_go_to_object_is_parked_with_the_map(tmp_path):
    from cloakos.app import Brain
    from cloakos.config import Config, load_config

    cfg = Config(load_config("config.yaml"))
    cfg["datalog"]["dir"] = str(tmp_path)
    cfg["voice"]["enabled"] = False
    cfg["llm"]["enabled"] = False
    brain = Brain(cfg)
    assert brain.mapper is None and await brain.go_to("mug-1") is False and brain.mode != "goto"


@pytest.mark.skip(reason="the 3D map is parked future work, so go-to-an-object is off")
async def test_go_to_drives_there_in_the_sim(tmp_path):
    pytest.importorskip("mujoco")
    from cloakos.app import Brain
    from cloakos.config import Config, load_config

    cfg = Config(load_config("config.yaml"))
    cfg["datalog"]["dir"] = str(tmp_path)
    cfg["voice"]["enabled"] = False
    cfg["llm"]["enabled"] = False
    cfg["map"] = {"enabled": False}
    cfg["sim"].update(enabled=True, camera=False, assets=False, world="empty", seed=5)
    brain = Brain(cfg)
    brain.skills.skills.clear()              # the simple controller, so this doesn't depend on training
    task = asyncio.create_task(brain.run(mode="auto", web=False))
    for _ in range(80):
        await asyncio.sleep(0.1)
        if brain.link.has_control:
            break
    sim = brain.link.sim
    sim.place_robot(0.0, 0.0, 0.0)
    brain.mapper.objects = [{"id": "mug-1", "label": "mug", "pos": [0.45, 0.2, 0.0], "seen": 3, "last_seen": 0}]
    seen = []
    brain.bus.on("goto.*", lambda ev: seen.append(ev.topic))
    r = await brain.command("go to the mug")
    assert r["action"] == "go_to" and brain.mode == "goto"
    for _ in range(300):
        await asyncio.sleep(0.1)
        if seen:
            break
    assert seen == ["goto.arrived"]
    p = sim.snapshot["pos"]
    assert math.dist(p[:2], (0.45, 0.2)) < 0.2
    await asyncio.sleep(2.5)
    assert brain.mode == "auto"                # back to what it was doing
    brain.request_stop()
    await asyncio.wait_for(task, 10)
    sim.stop()


def test_charging_rules_never_wander_off_half_charged():
    import types

    from cloakos.app import Brain

    modes = []

    class Fake:
        _charging_rules = Brain._charging_rules
        _switch_mode = Brain._switch_mode
        frozen = False
        mode = "auto"
        eyes = types.SimpleNamespace(flash=lambda *a: None)

        async def set_mode(self, m):
            modes.append(m)
            self.mode = m

        async def go_home(self):
            modes.append("go_home")

    async def run():
        b = Fake()
        b.world = w = types.SimpleNamespace(on_charger=False, charging=False, battery_level=2, battery_low=False)
        b._charging_rules(w)
        await asyncio.sleep(0)
        assert modes == []                              # driving around with a fine battery: leave it be
        w.on_charger, w.charging = True, True           # Vector docks itself
        b._charging_rules(w)
        await asyncio.sleep(0.01)
        assert modes == ["home"]
        b._charging_rules(w)
        await asyncio.sleep(0.01)
        assert modes == ["home"]                        # stays home while charging
        w.battery_level, w.charging = 3, False
        b._charging_rules(w)
        await asyncio.sleep(0.01)
        assert modes == ["home", "auto"]                # full: off it goes
        w.on_charger, w.battery_level, w.battery_low = False, 1, True
        b._charging_rules(w)
        await asyncio.sleep(0.01)
        assert modes[-1] == "go_home"

    asyncio.run(run())


async def test_volume_follows_wirepod_and_mutes_everything(tmp_path, monkeypatch):
    from cloakos.bus import Bus
    from cloakos.connection.robot import RobotLink
    from cloakos.world import World

    monkeypatch.setenv("CLOAK_ENV_FILE", str(tmp_path / ".env"))
    link = RobotLink({"robot": {"volume": 80}}, Bus(), World())
    link.target, link._serial = "real", "00e20100"
    calls = []

    class Client:
        async def master_volume(self, v):
            calls.append(("robot", v))

    async def fake(path, **params):
        calls.append((path, params))
        return '{"master_volume": 3}' if path.endswith("get_sdk_settings") else "done"

    link.client, link._wirepod = Client(), fake
    await link._pull_wirepod_volume()                 # changed in wire-pod's page -> slider follows
    assert link.volume == 50
    await link.set_volume(0)                          # slider to 0 -> a real mute on the robot and in wire-pod
    assert ("robot", 0) in calls and ("/api-sdk/volume", {"volume": 0}) in calls
    assert link._level(100) == 5 and link._level(10) == 1


async def test_teleop_camera_returns_to_the_same_angle():
    import asyncio
    import time

    from cloakos.teleop.policy import TeleopPolicy

    calls = []

    class Body:
        async def apply(self, a, source=None): pass
        async def head_angle(self, deg, speed=10.0): calls.append(("head", deg))
        async def lift_height(self, h, speed=10.0): calls.append(("lift", h))

    t = TeleopPolicy({"teleop": {"home_after_s": 0.3}}, Body())
    await t.start()
    await asyncio.sleep(0.05)
    assert ("head", -5.0) in calls and ("lift", 0.0) in calls
    calls.clear()
    t.input(head=1.0)
    await t.step(None)                                 # moving the head by hand
    t.input(head=0.0)
    await t.step(None)
    assert not calls                                   # not straight away
    time.sleep(0.35)
    t.input(throttle=0.5)
    await t.step(None)
    await asyncio.sleep(0.05)
    assert calls == [("head", -5.0), ("lift", 0.0)]    # back to the same view a moment later
