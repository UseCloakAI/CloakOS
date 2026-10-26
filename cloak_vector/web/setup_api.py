"""First-run setup wizard API: find wire-pod, pick the robot, choose a voice, connect a brain (LLM).

Answers are saved to .env next to config.yaml (the installer runs the app from its own
folder), then the app restarts itself to pick them up.
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, Response

if TYPE_CHECKING:
    from ..app import Brain

# Kokoro voices worth offering (id, label)
VOICES = [
    ("af_heart", "Heart · warm"), ("af_bella", "Bella · bright"), ("af_nicole", "Nicole · soft"),
    ("af_sky", "Sky · light"), ("am_adam", "Adam · low"), ("am_michael", "Michael · calm"),
    ("bf_emma", "Emma · British"), ("bm_george", "George · British"),
]
POD_GUESSES = ["http://escapepod.local:8080", "http://localhost:8080", "http://127.0.0.1:8080"]
ENV_KEYS = {"WIREPOD_URL", "VECTOR_SERIAL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "KOKORO_VOICE", "CLOAK_SETUP_DONE"}


def env_path() -> Path:
    return Path(os.environ.get("CLOAK_ENV_FILE", ".env"))


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def write_env(path: Path, updates: dict[str, str]) -> None:
    """Update keys in place, keep comments and unrelated lines."""
    lines = path.read_text().splitlines() if path.exists() else []
    done = set()
    for i, line in enumerate(lines):
        m = re.match(r"\s*#?\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in updates and m.group(1) not in done:
            lines[i] = f"{m.group(1)}={updates[m.group(1)]}"
            done.add(m.group(1))
    lines += [f"{k}={v}" for k, v in updates.items() if k not in done]
    path.write_text("\n".join(lines) + "\n")
    try:
        path.chmod(0o600)          # it can hold an API key
    except OSError:
        pass


async def probe_pod(url: str, http: httpx.AsyncClient) -> dict[str, Any] | None:
    url = url.strip().rstrip("/")
    if not url:
        return None
    if not url.startswith("http"):
        url = "http://" + url
    if not re.search(r":\d+$", url):
        url += ":8080"
    try:
        info = (await http.get(f"{url}/api-sdk/get_sdk_info")).json()
    except Exception:
        return None
    bots = [{"serial": b.get("esn"), "ip": b.get("ip_address"), "ready": bool(b.get("guid") or info.get("global_guid"))}
            for b in info.get("robots") or []]
    return {"url": url, "bots": bots}


def make_router(brain: "Brain") -> APIRouter:
    r = APIRouter(prefix="/api/setup")

    @r.get("/status")
    async def status() -> dict[str, Any]:
        env = read_env(env_path())
        return {
            "done": env.get("CLOAK_SETUP_DONE") == "1",
            "wirepod": env.get("WIREPOD_URL", ""),
            "serial": env.get("VECTOR_SERIAL", ""),
            "llm": {"base_url": env.get("LLM_BASE_URL", ""), "model": env.get("LLM_MODEL", ""), "has_key": bool(env.get("LLM_API_KEY"))},
            "voice": env.get("KOKORO_VOICE", "af_heart"),
            "voices": [{"id": v, "label": label} for v, label in VOICES],
            "tts_ready": brain.voice.tts is not None,
        }

    @r.post("/scan")
    async def scan(body: dict = Body(default={})) -> dict[str, Any]:
        urls = [body["url"]] if body.get("url") else POD_GUESSES
        async with httpx.AsyncClient(timeout=2.5) as http:
            found = await asyncio.gather(*(probe_pod(u, http) for u in urls))
        hits = [f for f in found if f]
        return {"found": hits[0] if hits else None, "tried": urls}

    @r.post("/llm")
    async def llm_test(body: dict = Body(...)) -> dict[str, Any]:
        base = (body.get("base_url") or "").rstrip("/")
        headers = {"Authorization": f"Bearer {body['api_key']}"} if body.get("api_key") else {}
        req = {"model": body.get("model", ""), "max_tokens": 8,
               "messages": [{"role": "user", "content": "Say hi in two words."}]}
        try:
            async with httpx.AsyncClient(timeout=15, headers=headers) as http:
                resp = await http.post(f"{base}/chat/completions", json=req)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
            return {"ok": True, "reply": (text or "").strip()[:80]}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}

    @r.get("/voice")
    async def voice_preview(voice: str = "af_heart", text: str = "Hi! I'm awake.") -> Response:
        tts = brain.voice.tts
        if tts is None or not hasattr(tts, "_k"):
            return JSONResponse({"error": brain.voice.tts_error or "voice model not loaded"}, status_code=503)
        samples, sr = await asyncio.to_thread(tts._k.create, text[:80], voice=voice, speed=tts.speed, lang=tts.lang)
        import numpy as np
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        return Response(buf.getvalue(), media_type="audio/wav")

    @r.post("/save")
    async def save(body: dict = Body(...)) -> dict[str, Any]:
        updates = {"CLOAK_SETUP_DONE": "1"}
        for field, key in (("wirepod", "WIREPOD_URL"), ("serial", "VECTOR_SERIAL"), ("voice", "KOKORO_VOICE"),
                           ("llm_base_url", "LLM_BASE_URL"), ("llm_model", "LLM_MODEL")):
            if body.get(field) is not None:
                updates[key] = str(body[field]).strip()
        if body.get("llm_api_key"):
            updates["LLM_API_KEY"] = str(body["llm_api_key"]).strip()
        write_env(env_path(), {k: v for k, v in updates.items() if k in ENV_KEYS})
        asyncio.get_running_loop().call_later(0.5, brain.request_restart)
        return {"ok": True}

    return r
