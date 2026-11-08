"""Simulator endpoints: switch robot/sim, reset worlds, poke the robot, run tests, 3D viewer feed."""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, Response, WebSocket, WebSocketDisconnect

from ..sim.rooms import OBJECTS, WORLDS
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

    def deskrun():
        if getattr(brain, "deskrun", None) is None:
            from ..sim.deskrun import DeskRun
            brain.deskrun = DeskRun(brain)
        return brain.deskrun

    @r.get("/desk")
    async def desk() -> dict[str, Any]:
        sim = brain.link.sim
        pols = [p["name"] for p in await asyncio.to_thread(brain.trainer.policies)
                if p["ready"] and (p["task"] or "").startswith("desk")]
        return {**(sim.desk_layout() if sim else {"size": None, "objects": []}), "catalog": {k: v[2][:3] for k, v in OBJECTS.items()},
                "policies": pols, "run": deskrun().status()}

    @r.post("/desk")
    async def desk_edit(body: dict = Body(...)) -> dict[str, Any]:
        """size: [w_cm, d_cm]; add: {name, x?, y?}; remove: id; clear: true; random: true."""
        if brain.link.target != "sim":
            await brain.link.set_target("sim")
        sim = brain.link.get_sim()
        if body.get("random"):
            sim.layout = None
            await asyncio.to_thread(sim.load, "desk", None)
            return {"ok": True}
        cur = sim.desk_layout()
        objs = [{"name": o["name"], "x": o["x"], "y": o["y"]} for o in cur["objects"] if not o["fell"]]
        size = cur["size"] or [80, 50]
        if body.get("size"):
            size = [max(20, min(300, float(v))) for v in body["size"]][:2]
        if body.get("clear"):
            objs = []
        if body.get("remove"):
            objs = [{"name": o["name"], "x": o["x"], "y": o["y"]} for o in cur["objects"]
                    if o["id"] != body["remove"] and not o["fell"]]
        add = body.get("add")
        if add and add.get("name") in OBJECTS:
            objs.append({"name": add["name"], "x": add.get("x"), "y": add.get("y")})
        await asyncio.to_thread(sim.set_layout, {"size": size, "objects": objs})
        return {"ok": True}

    @r.post("/desk/run")
    async def desk_run(body: dict = Body(...)) -> dict[str, Any]:
        why = await deskrun().start(str(body.get("policy", "")))
        return {"ok": why is None, "error": why}

    @r.post("/desk/stop")
    async def desk_stop() -> dict[str, Any]:
        await deskrun().stop()
        return {"ok": True}

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
        sim.layout = None                               # New room: a random desk again
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

    @r.get("/mesh/{mesh}")
    async def mesh(mesh: int) -> Response:
        sim = brain.link.sim
        data = await asyncio.to_thread(sim.mesh_data, mesh) if sim else None
        return Response(data, media_type="application/octet-stream") if data else Response(status_code=404)

    @r.get("/texture/{tex}")
    async def texture(tex: int) -> Response:
        sim = brain.link.sim
        data = await asyncio.to_thread(sim.texture_png, tex) if sim else None
        return Response(data, media_type="image/png") if data else Response(status_code=404)

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
                sim.touch()
                snap = sim.snapshot or {}
                msg = {"v": sim.version, "p": sim.poses(), "frozen": brain.frozen, "goal": brain.learned.goal,
                       "s": {k: snap.get(k) for k in ("upright", "pitch", "speed", "picked_up", "cliff", "bump")},
                       "test": t.running if t else None}
                await ws.send_text(json.dumps(msg))
            await asyncio.sleep(1 / 30)
    except (WebSocketDisconnect, RuntimeError):
        pass
