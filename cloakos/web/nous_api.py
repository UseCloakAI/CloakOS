"""Nous model endpoints: which version is loaded, and the Update button."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

from .. import NOUS_VERSION, nous

if TYPE_CHECKING:
    from ..app import Brain


def make_nous_router(brain: "Brain") -> APIRouter:
    r = APIRouter(prefix="/api/nous")

    def local() -> dict[str, Any]:
        return {"installed": nous.installed(), "build": NOUS_VERSION, "skills": brain.skills.summary()}

    @r.get("")
    async def status() -> dict[str, Any]:
        return local()

    @r.post("/check")
    async def check() -> dict[str, Any]:
        info = await asyncio.to_thread(nous.check)
        info.pop("manifest", None)
        return {**local(), **info}

    @r.post("/update")
    async def update() -> dict[str, Any]:
        info = await asyncio.to_thread(nous.update)
        if info.get("updated"):
            brain.skills.reload()
        info.pop("manifest", None)
        return {**local(), **info}

    return r
