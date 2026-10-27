"""Dashboard: FastAPI on the same event loop as the brain.

  GET  /             the UI (Cloak chat styling, vendored cloak.css)
  GET  /camera.mjpg  live camera
  WS   /ws           server -> {type:"state"} at 5 Hz + {type:"event"} as they happen
                     client -> mode / teleop / say / talk / anim / record / poke / simulate
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
from urllib.parse import quote
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..bus import Event
from .. import FULL_NAME, __version__
from ..drives import DRIVES
from .setup_api import autoconfigure_loop, env_path, make_router, robot_ota_url, write_env
from .sim_api import make_sim_router, sim_ws

if TYPE_CHECKING:
    from ..app import Brain

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
# events worth showing in the UI stream
SHOWN = ("sense.", "drive.", "voice.", "behavior.", "mode.", "robot.", "record.")
SIM_EVENTS = ("face_seen", "touched", "picked_up", "put_down", "cliff", "loud_sound", "heard_wake_word")


def create_app(brain: "Brain") -> FastAPI:
    app = FastAPI(title=FULL_NAME, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def revalidate(request: Request, call_next):
        # the UI changes with every update: always revalidate (cheap 304s via ETag), never serve stale files
        resp = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/") or path in ("/setup", "/pair"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp
    clients: set[WebSocket] = set()
    history: list[dict[str, Any]] = []

    def on_event(ev: Event) -> None:
        if not ev.topic.startswith(SHOWN):
            return
        msg = {"type": "event", "topic": ev.topic, "t": ev.t,
               "data": {k: v for k, v in ev.data.items() if k != "scores"}}
        history.append(msg)
        del history[:-200]
        text = json.dumps(msg, default=str)
        for ws in list(clients):
            asyncio.ensure_future(_send(ws, text))

    async def _send(ws: WebSocket, text: str) -> None:
        try:
            await ws.send_text(text)
        except Exception:
            clients.discard(ws)

    brain.bus.on("*", on_event)

    app.include_router(make_router(brain))
    app.include_router(make_sim_router(brain))

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/setup")
    async def setup_page() -> FileResponse:
        return FileResponse(STATIC / "setup.html")

    @app.get("/pair", response_model=None)
    async def pair_page(request: Request) -> Response:
        # the pairing code reads its firmware URL from the query string
        if "otaUrl" not in request.query_params:
            pod = (brain.cfg.get("wirepod", {}) or {}).get("url") or ""
            return RedirectResponse(f"/pair?otaUrl={quote(robot_ota_url(pod), safe='')}")
        return FileResponse(STATIC / "pair.html")

    @app.get("/api/info")
    async def info() -> JSONResponse:
        c = brain.cfg
        bot = brain.link.client.bot if brain.link.client else None
        return JSONResponse({
            "product": {"name": FULL_NAME, "version": __version__},
            "wirepod": (c.get("wirepod", {}) or {}).get("url"),
            "robot": {"name": bot.name if bot else None, "serial": bot.serial if bot else None,
                      "host": bot.host if bot else None},
            "emotes": brain.body.emotes.names if brain.body.emotes else [],
            "llm": {"base_url": (c.get("llm", {}) or {}).get("base_url"), "model": (c.get("llm", {}) or {}).get("model")},
            "tts": brain.voice.tts.id if brain.voice.tts else None,
            "tts_error": brain.voice.tts_error,
            "anim_triggers": brain.link.anim_triggers,
            "episodes": brain.recorder.list_episodes(),
            "drives": list(DRIVES),
            "sim_events": list(SIM_EVENTS),
        })

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "mode": brain.mode, "recording": brain.recorder.active, "connection": brain.link.state}

    @app.get("/eyes.png")
    async def eyes() -> Response:
        img = brain.eye_frame
        if img is None:
            return Response(status_code=204)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/camera.mjpg")
    async def camera() -> StreamingResponse:
        async def frames():
            last = -1
            while True:
                w = brain.world
                if w.frame is not None and w.frame_id != last:
                    last = w.frame_id
                    buf = io.BytesIO()
                    await asyncio.to_thread(lambda: w.frame.convert("RGB").save(buf, "JPEG", quality=70))
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
                await asyncio.sleep(0.08)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.delete("/api/episodes/{name}")
    async def delete_episode(name: str) -> JSONResponse:
        ok = await asyncio.to_thread(brain.recorder.delete, name)
        return JSONResponse({"ok": ok}, status_code=200 if ok else 404)

    @app.websocket("/ws/sim")
    async def ws_sim(ws: WebSocket) -> None:
        await sim_ws(brain, ws)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_text(json.dumps({"type": "history", "events": history}, default=str))
        clients.add(ws)
        pusher = asyncio.create_task(_push_state(ws))
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                await handle(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            clients.discard(ws)
            pusher.cancel()

    async def _push_state(ws: WebSocket) -> None:
        while True:
            try:
                await ws.send_text(json.dumps({"type": "state", **brain.state()}, default=str))
            except Exception:
                return
            await asyncio.sleep(0.2)

    async def handle(msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "mode":
            await brain.set_mode(msg.get("mode", ""))
        elif kind == "teleop":
            brain.teleop.input(**{k: msg[k] for k in ("throttle", "steer", "head", "lift", "boost") if k in msg})
        elif kind == "say":
            asyncio.ensure_future(brain.voice.say(str(msg.get("text", ""))[:200], trigger="manual", source="you"))
        elif kind == "talk":
            asyncio.ensure_future(_talk(str(msg.get("text", ""))[:500]))
        elif kind == "volume":
            await brain.link.set_volume(int(msg.get("value", 80)))
            write_env(env_path(), {"CLOAK_VOLUME": str(brain.link.volume)})
        elif kind == "freeze":
            await brain.set_frozen(bool(msg.get("on")), source="ui")
        elif kind == "anim":
            asyncio.ensure_future(brain.body.anim(str(msg.get("key", ""))))
        elif kind == "emote":
            brain.react(str(msg.get("name", "")))
        elif kind == "record":
            brain.toggle_recording(bool(msg.get("on")), note=str(msg.get("note", "")))
        elif kind == "poke" and msg.get("drive") in DRIVES:
            brain._emit_crossings(brain.drives.set(msg["drive"], float(msg.get("value", 0.5))))
        elif kind == "simulate" and msg.get("event") in SIM_EVENTS:
            brain.bus.emit(f"sense.{msg['event']}", simulated=True)

    async def _talk(text: str) -> None:
        reply = await brain.voice.reply(text)
        if not reply:
            brain.bus.emit("voice.failed", text=text, reason="no reply from LLM" if brain.voice.llm.enabled else "LLM not configured")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


async def serve(brain: "Brain") -> None:
    import uvicorn

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):   # the brain owns Ctrl+C
            yield

        def install_signal_handlers(self) -> None:
            pass

    c = brain.cfg.get("web", {}) or {}
    host, port = c.get("host", "127.0.0.1"), int(c.get("port", 8765))
    server = Server(uvicorn.Config(create_app(brain), host=host, port=port, log_level="warning", lifespan="off"))
    log.info("dashboard: http://%s:%d", host, port)
    auto = asyncio.ensure_future(autoconfigure_loop(brain))
    try:
        await server.serve()
    except asyncio.CancelledError:
        server.should_exit = True
        raise
    finally:
        auto.cancel()
