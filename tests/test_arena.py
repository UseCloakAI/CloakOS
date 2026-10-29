import asyncio


async def test_arena_scores_a_real_run_and_feeds_desk_size(tmp_path, monkeypatch):
    from cloakos.arena.session import Arena

    monkeypatch.setenv("CLOAK_ENV_FILE", str(tmp_path / ".env"))

    class Link:
        target, edge_events = "real", 0

    class Trainer:
        root = tmp_path / "policies"

    class Brain:
        estop, mode, link, trainer = False, "teleop", Link(), Trainer()

    a = Arena(Brain(), tmp_path)
    a.save_config({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "size": [80, 50]})
    assert (tmp_path / ".env").read_text().strip().endswith("CLOAK_DESK=80x50")
    a.update_pose({"ok": True, "x": 0.1, "y": 0.1, "yaw": 0.0})
    assert await a.start(None, 30) is None
    for i in range(10):                                   # drives 50 cm along the desk
        a.update_pose({"ok": True, "x": 0.1 + i * 0.05, "y": 0.1, "yaw": 0.0})
        Brain.link.edge_events = i // 4
        await asyncio.sleep(0.12)
    a.update_pose({"ok": True, "x": 0.9, "y": 0.1, "yaw": 0.0})   # past the right edge: off the desk
    await asyncio.sleep(0.3)
    r = a.summary()
    assert r["status"] == "fell" and r["falls"] == 1 and r["cells"] >= 8 and r["edge_saves"] == 2
    assert a.recent()[0]["status"] == "fell" and not a.running


async def test_routes_to_the_charger_docks_and_comes_back(tmp_path, monkeypatch):
    import math
    import time

    from cloakos.arena.session import Arena

    monkeypatch.setenv("CLOAK_ENV_FILE", str(tmp_path / ".env"))

    class World:
        pose, on_charger, battery_level, charging, battery_low = (0.0, 0.0, 0.0), False, 1, False, True

    class Link:
        target, edge_events, connected = "real", 0, True

    class Trainer:
        root = tmp_path / "policies"

    class Body:
        def __init__(self, bot): self.bot = bot

        async def drive(self, l, r, source=None): self.bot["q"].append((time.time() + 0.4, (l, r)))   # lands 0.4 s late

        async def leave_home(self):
            w.on_charger = False
            self.bot["left"] = True

    w = World()
    bot = {"x": 0.3, "y": 0.15, "th": math.pi / 2, "v": (0.0, 0.0), "odo0": 1.0, "q": []}   # odometry yaw is offset by 1 rad

    class Brain:
        estop, frozen, mode, link, trainer, world = False, False, "auto", Link(), Trainer(), w

        async def set_mode(self, m): Brain.mode = m

        async def go_home(self):
            await self.set_mode("home")
            t0 = time.time()
            ok = await a.route_to_charger(timeout=40)
            bot["took"] = time.time() - t0
            bot["arrived"] = ok
            w.on_charger, w.charging = True, True
            return ok

    br = Brain()
    br.body = Body(bot)
    a = Arena(br, tmp_path)
    a.save_config({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "size": [140, 60], "charger": {"x": 1.2, "y": 0.45, "yaw": math.pi}})

    async def physics():                              # the robot moves; the camera and odometry see it
        while True:
            while bot["q"] and bot["q"][0][0] <= time.time():
                bot["v"] = bot["q"].pop(0)[1]
            l, r = bot["v"]
            v, om = (l + r) / 2 / 1000, (r - l) / 1000 / 0.048
            bot["x"] += v * math.cos(bot["th"]) * 0.05
            bot["y"] += v * math.sin(bot["th"]) * 0.05
            bot["th"] += om * 0.05
            w.pose = (bot["x"] * 1000, bot["y"] * 1000, bot["th"] - bot["odo0"])
            moving = abs(v) > 0.01
            a.update_pose({"ok": True, "x": bot["x"], "y": bot["y"], "yaw": bot["th"] if moving else 0.0,
                           "yaw_src": "motion" if moving else "hold"})
            await asyncio.sleep(0.05)


    sim = asyncio.create_task(physics())
    a.update_pose({"ok": True, "x": 0.3, "y": 0.15, "yaw": math.pi / 2, "yaw_src": "light"})
    assert a.can_route() and a.dock() is None
    for _ in range(600):
        await asyncio.sleep(0.1)
        if a.dock_state == "charging":
            break
    assert bot.get("arrived"), "never reached the approach point"
    print("route took", round(bot["took"], 1), "s")
    assert bot["took"] < 20                                                         # ~1 m with 0.4 s lag: no zigzagging
    assert math.hypot(bot["x"] - 1.02, bot["y"] - 0.45) < 0.06                      # 18 cm in front of the charger
    assert abs(math.atan2(math.sin(bot["th"] - math.pi), math.cos(bot["th"] - math.pi))) > 2.9   # facing it (yaw ~ 0)
    w.battery_level, w.charging = 3, False             # charged
    for _ in range(100):
        await asyncio.sleep(0.1)
        if a.dock_state is None:
            break
    assert bot.get("left") and Brain.mode == "auto"
    sim.cancel()


async def test_arena_page_puts_vector_in_teleop_and_back(tmp_path, monkeypatch):
    from cloakos.arena.session import Arena

    monkeypatch.setenv("CLOAK_ENV_FILE", str(tmp_path / ".env"))

    class Brain:
        estop, mode = False, "auto"

        async def set_mode(self, m): Brain.mode = m

    a = Arena(Brain(), tmp_path)
    assert await a.enter() == "teleop"
    assert await a.leave() == "auto"
    Brain.mode = "home"                                  # charging: leave it alone
    assert await a.enter() == "home" and await a.leave() == "home"


async def test_a_run_can_start_from_the_charger(tmp_path, monkeypatch):
    from cloakos.arena.session import Arena

    monkeypatch.setenv("CLOAK_ENV_FILE", str(tmp_path / ".env"))

    class World:
        on_charger, battery_low, pose = True, False, None

    class Link:
        target, edge_events = "real", 0

    class Trainer:
        root = tmp_path / "policies"

    class Body:
        async def leave_home(self):
            World.on_charger = False
            a.update_pose({"ok": True, "x": 0.5, "y": 0.3, "yaw": 0.0})

    class Brain:
        estop, mode, link, trainer, world, body = False, "home", Link(), Trainer(), World(), Body()

        async def set_mode(self, m): Brain.mode = m

    a = Arena(Brain(), tmp_path)
    a.save_config({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "size": [140, 60]})
    assert await a.start(None, 30) is None
    assert not World.on_charger and Brain.mode == "teleop" and a.running
    await a.stop()
