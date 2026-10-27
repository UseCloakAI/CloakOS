"""Simulator endpoints: switch robot/sim, reset worlds, poke the robot, run tests, 3D viewer feed."""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect

from ..sim.rooms import WORLDS
from .setup_api import env_path, write_env

if TYPE_CHECKING:
    from ..app import Brain

POKES = {"knock", "pick_up", "tap_button"}


def make_sim_router(brain: "Brain") -> APIRouter:
    r = APIRouter(prefix="/api/sim")

    def tester():
        if brain.tester is None:
            from ..sim.tester import Tester
            brain.tester = Tester(brain)
        return brain.tester

    @r.get("")
    async def status() -> dict[str, Any]:
        sim = brain.link.sim
        t = tester()
        return {
            "target": brain.link.target, "worlds": WORLDS,
            "world": sim.world if sim else None, "seed": sim.seed if sim else None,
            "render_error": sim.render_error if sim else None,
            "scenarios": t.catalog(), "running": t.running, "results": list(reversed(t.results))[:20],
        }

    @r.post("/target")
    async def target(body: dict = Body(...)) -> dict[str, Any]:
        await brain.link.set_target(str(body.get("target", "real")))
        write_env(env_path(), {"CLOAK_SIM": "1" if brain.link.target == "sim" else "0"})
        return {"target": brain.link.target}

    @r.post("/reset")
    async def reset(body: dict = Body(default={})) -> dict[str, Any]:
        sim = brain.link.get_sim()
        seed = body.get("seed")
        sc = await asyncio.to_thread(sim.load, body.get("world") or sim.world, int(seed) if seed not in (None, "") else None)
        return {"world": sc.world, "seed": sc.seed}

    @r.post("/poke")
    async def poke(body: dict = Body(...)) -> dict[str, Any]:
        sim = brain.link.sim
        action = body.get("action")
        if sim is None or action not in POKES:
            return {"ok": False}
        getattr(sim, action)(**({"n": 3} if action == "tap_button" and body.get("triple") else {}))
        return {"ok": True}

    @r.post("/test")
    async def run_test(body: dict = Body(...)) -> dict[str, Any]:
        seed = body.get("seed")
        ok = tester().start(str(body.get("scenario")), int(seed) if seed not in (None, "") else None, bool(body.get("record")))
        return {"ok": ok}

    @r.post("/test/stop")
    async def stop_test() -> dict[str, Any]:
        tester().stop()
        return {"ok": True}

    @r.get("/scene")
    async def scene() -> dict[str, Any]:
        sim = brain.link.sim
        return sim.scene() if sim else {"version": 0, "geoms": []}

    return r


async def sim_ws(brain: "Brain", ws: WebSocket) -> None:
    """3D viewer feed: body poses at ~30 Hz, plus the scene version so the viewer reloads on a new world."""
    await ws.accept()
    try:
        while True:
            sim = brain.link.sim
            t = brain.tester
            if sim is not None and sim.scene_info is not None:
                snap = sim.snapshot or {}
                msg = {"v": sim.version, "p": sim.poses(), "frozen": brain.frozen,
                       "s": {k: snap.get(k) for k in ("upright", "pitch", "speed", "picked_up", "cliff", "bump")},
                       "test": t.running if t else None}
                await ws.send_text(json.dumps(msg))
            await asyncio.sleep(1 / 30)
    except (WebSocketDisconnect, RuntimeError):
        pass
