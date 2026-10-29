"""Arena endpoints: desk setup, the pose stream from the page's camera, scored real runs."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body

if TYPE_CHECKING:
    from ..app import Brain


def make_arena_router(brain: "Brain") -> APIRouter:
    r = APIRouter(prefix="/api/arena")

    def arena():
        if getattr(brain, "arena", None) is None:
            from ..arena.session import Arena
            brain.arena = Arena(brain, brain.trainer.root.parent)     # next to policies/ (the data dir)
        return brain.arena

    @r.get("")
    async def status() -> dict[str, Any]:
        a = arena()
        pols = [p["name"] for p in await asyncio.to_thread(brain.trainer.policies)
                if p["ready"] and p["task"] == "desk-pixels"]
        return {"config": a.config, "pose": a.pose, "run": a.summary(), "running": a.running, "dock": a.dock_state, "route": a.route,
                "battery": brain.world.battery_level, "on_charger": brain.world.on_charger,
                "recent": await asyncio.to_thread(a.recent), "policies": pols,
                "target": brain.link.target, "mode": brain.mode}

    @r.post("/config")
    async def config(body: dict = Body(...)) -> dict[str, Any]:
        return {"ok": True, "config": await asyncio.to_thread(arena().save_config, body)}

    @r.post("/pose")
    async def pose(body: dict = Body(...)) -> dict[str, Any]:
        arena().update_pose(body)
        return {"ok": True}

    @r.post("/run")
    async def run(body: dict = Body(...)) -> dict[str, Any]:
        why = await arena().start(body.get("policy") or None, float(body.get("seconds", 120)))
        return {"ok": why is None, "error": why}

    @r.post("/enter")
    async def enter() -> dict[str, Any]:
        return {"mode": await arena().enter()}

    @r.post("/leave")
    async def leave() -> dict[str, Any]:
        return {"mode": await arena().leave()}

    @r.post("/dock")
    async def dock() -> dict[str, Any]:
        why = arena().dock()
        return {"ok": why is None, "error": why}

    @r.post("/run/stop")
    async def stop() -> dict[str, Any]:
        await arena().stop()
        return {"ok": True}

    return r
