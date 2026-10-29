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
