"""Reinforcement learning endpoints: train in the background, chart progress, use a policy."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from ..rl.envs import SENSORS, TASKS
from ..rl.manager import available

if TYPE_CHECKING:
    from ..app import Brain


def make_rl_router(brain: "Brain") -> APIRouter:
    r = APIRouter(prefix="/api/rl")
    t = brain.trainer

    @r.get("")
    async def status() -> dict[str, Any]:
        lp = brain.learned
        return {"available": available(), "tasks": TASKS, "sensors": SENSORS, "running": t.running(),
                "policies": await asyncio.to_thread(t.policies),
                "active": {"name": lp.policy_name, "task": lp.spec.task if lp.spec else None, "goal": lp.goal,
                           "status": lp.status, "on": brain.mode == "learned", "blocked": lp.deployable()}}

    @r.get("/calibration")
    async def calibration() -> dict[str, Any]:
        c = brain.calibrator
        return {"running": c.running, "kind": c.kind, "progress": c.progress, "result": c.result}

    @r.post("/calibration")
    async def calibrate(body: dict = Body(...)) -> dict[str, Any]:
        why = brain.calibrator.start(str(body.get("kind")), float(body.get("seconds", 180)))
        return {"ok": why is None, "error": why}

    @r.post("/calibration/send")
    async def calibration_send() -> dict[str, Any]:
        from .. import share
        c = brain.calibrator
        if not c.result:
            return {"ok": False, "error": "Run a calibration first."}
        try:
            await asyncio.to_thread(share.upload_calibration, c.result, brain.link.target)
        except Exception as e:
            return {"ok": False, "error": f"Couldn't send: {e}"}
        return {"ok": True}

    @r.post("/calibration/stop")
    async def calibration_stop() -> dict[str, Any]:
        if brain.calibrator.task:
            brain.calibrator.task.cancel()
        return {"ok": True}

    @r.post("/train")
    async def train(body: dict = Body(...)) -> JSONResponse:
        try:
            name = t.start(str(body.get("task", "goto")), str(body.get("sensors", "robot")),
                           int(body.get("steps", 300_000)), body.get("envs"), body.get("resume"))
        except (RuntimeError, ValueError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "name": name})

    @r.post("/stop")
    async def stop() -> dict[str, Any]:
        await asyncio.to_thread(t.stop)
        return {"ok": True}

    @r.get("/progress/{name}")
    async def progress(name: str) -> list[dict]:
        return await asyncio.to_thread(t.progress, name)

    @r.post("/use")
    async def use(body: dict = Body(...)) -> JSONResponse:
        name = str(body.get("name", ""))
        try:
            await asyncio.to_thread(brain.learned.load, t.root / name)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"couldn't load {name}: {e}"}, status_code=400)
        if brain.mode == "learned":
            await brain.set_mode("idle")
        await brain.set_mode("learned")
        return JSONResponse({"ok": True, "blocked": brain.learned.deployable()})

    @r.post("/goal")
    async def goal(body: dict = Body(...)) -> dict[str, Any]:
        x, y = body.get("x"), body.get("y")
        brain.learned.goal = (float(x), float(y)) if x is not None and y is not None else None
        return {"goal": brain.learned.goal}

    @r.delete("/{name}")
    async def delete(name: str) -> JSONResponse:
        ok = await asyncio.to_thread(t.delete, name)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 400)

    return r
