"""3D map endpoints: what Vector has mapped, the objects it remembers, and "go there"."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, Response

if TYPE_CHECKING:
    from ..app import Brain


def make_map_router(brain: "Brain") -> APIRouter:
    r = APIRouter(prefix="/api/map")
    mp = brain.mapper

    @r.get("")
    async def summary() -> dict[str, Any]:
        out = mp.summary()
        sim = brain.link.sim
        if brain.link.target == "sim" and sim is not None and sim.snapshot:
            s = sim.snapshot
            out["robot"] = {"pos": s["pos"], "yaw": s["yaw"]}
        elif brain.link.pose:
            p = brain.link.pose
            out["robot"] = {"pos": p["pos"], "yaw": brain.world.pose[2] if brain.world.pose else 0.0}
        out["goal"] = brain.learned.goal
        return out

    @r.get("/voxels")
    async def voxels() -> Response:
        return Response(await asyncio.to_thread(mp.voxels_bin), media_type="application/octet-stream")

    @r.post("/clear")
    async def clear() -> dict[str, Any]:
        await asyncio.to_thread(mp.clear)
        return {"ok": True}

    @r.post("/enabled")
    async def enabled(body: dict = Body(...)) -> dict[str, Any]:
        mp.enabled = bool(body.get("on", True))
        return {"enabled": mp.enabled}

    @r.post("/goto")
    async def goto(body: dict = Body(...)) -> dict[str, Any]:
        """Drive to a remembered object (by id) or the best match for a word ("mug")."""
        o = next((o for o in mp.objects if o["id"] == body.get("id")), None) or mp.find(str(body.get("query", "")))
        if o is None:
            return {"ok": False, "error": "I haven't seen one of those yet."}
        brain.learned.goal = (o["pos"][0], o["pos"][1])
        lp = brain.learned
        if lp.spec is None or lp.spec.task != "goto":
            return {"ok": True, "object": o, "note": "Goal set. Pick a trained goto policy in Training to drive there."}
        if brain.mode != "learned":
            await brain.set_mode("learned")
        return {"ok": True, "object": o, "blocked": lp.deployable()}

    return r
