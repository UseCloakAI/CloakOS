import time
import types

from cloakos.expression.eyes import EyeShape, Eyes
from cloakos.expression.sfx import EVENTS, SOUNDS, Sfx, render


def test_every_sound_renders_and_follows_the_mood():
    for name in SOUNDS:
        pcm = render(name)
        assert len(pcm) % 2 == 0 and 0.03 < len(pcm) / 2 / 16000 < 2.0, name
    calm = len(render("go", {"energy": 0.9, "fear": 0, "boredom": 0}))
    bored = len(render("go", {"energy": 0.1, "fear": 0, "boredom": 0.9}))
    assert bored > calm                                     # bored = slower


def test_sfx_cooldowns_and_quiet_while_talking():
    played = []

    class Bus:
        def on(self, *a): pass

    async def play_pcm(pcm, vol): played.append(len(pcm))
    body = types.SimpleNamespace(ready=True, speaking_until=0.0, play_pcm=play_pcm)
    s = Sfx({}, Bus(), body, types.SimpleNamespace(values={"energy": 0.5}))
    import asyncio

    async def go():
        assert s.play("pickup", 1.0)
        assert not s.play("pickup", 1.0)                    # cooldown
        s._any = 0
        body.speaking_until = time.time() + 5
        assert not s.play("found", 0.0)                     # talking
        await asyncio.sleep(0)
    asyncio.run(go())
    assert played and set(EVENTS.values()) and all(v[0] in SOUNDS for v in EVENTS.values())


def test_eyes_tween_and_show_status():
    e = Eyes({})
    t = 100.0
    e._last = t
    target = EyeShape(gaze_x=1.0)
    xs = []
    for i in range(30):
        t += 0.05
        e.frame(target, now=t)
        xs.append(e.live.gaze_x)
    assert 0 < xs[0] < 0.6 and abs(xs[-1] - 1.0) < 0.05     # eases over, doesn't jump
    e.status = "Wandering"
    for _ in range(20):
        t += 0.05
        img = e.frame(target, now=t)
    assert e._ov > 0.9 and img.size == (184, 96)
    lit = sum(1 for p in img.getdata() if p[0] > 150)
    assert lit > 1500                                       # big words fill the screen
    e.status = None
    e.flash("Got there!", 1.0)
    assert e._text(t) == "Got there!"


def test_new_activity_takes_the_screen_then_eyes_return():
    e = Eyes({})
    e._last = 0.0
    e.status = "Wandering"
    seen = []
    t = 0.0
    for _ in range(60):
        t += 0.05
        e.frame(EyeShape(), now=t)
        seen.append(e._ov)
    assert max(seen) == 1.0 and seen[-1] == 0.0          # words, then back to eyes while still wandering
    e.status = "Going to the mug"                        # a new activity is announced again
    t += 0.05
    e.frame(EyeShape(), now=t)
    t += 0.3
    e.frame(EyeShape(), now=t)
    assert e._ov > 0.9
