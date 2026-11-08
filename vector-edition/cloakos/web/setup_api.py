"""First-run setup wizard API: find wire-pod, pick the robot, choose a voice, connect a brain (LLM).

Answers are saved to .env next to config.yaml (the installer runs the app from its own
folder), then the app restarts itself to pick them up.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import subprocess
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
ENV_KEYS = {"CLOAK_AUTO_UPDATE", "WIREPOD_URL", "VECTOR_SERIAL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "KOKORO_VOICE", "CLOAK_SETUP_DONE", "CLOAK_VOLUME", "CLOAK_SIM"}


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
        resp = await http.get(f"{url}/api-sdk/get_sdk_info")
        if resp.status_code != 200:
            # a fresh wire-pod answers 500 ("no bots are authenticated") but it's there
            running = (await http.get(f"{url}/api/is_running")).text.strip() == "true"
            return {"url": url, "bots": []} if running else None
        info = resp.json()
    except Exception:
        return None
    bots = [{"serial": b.get("esn"), "ip": b.get("ip_address"), "ready": bool(b.get("guid") or info.get("global_guid"))}
            for b in info.get("robots") or []]
    return {"url": url, "bots": bots}


def cloak_home() -> Path | None:
    h = os.environ.get("CLOAK_HOME")
    return Path(h) if h else None


def update_info() -> dict[str, Any]:
    """What the installed app knows about updates (empty when running from a dev checkout)."""
    home = cloak_home()
    if home is None:
        return {"managed": False}
    def text(name: str) -> str:
        p = home / name
        return p.read_text().strip() if p.exists() else ""
    status: dict[str, Any] = {}
    try:
        status = json.loads(text("update-status.json") or "{}")
    except ValueError:
        pass
    compat: dict[str, Any] = {}
    try:
        compat = json.loads(text("app/compat.json") or "{}")
    except ValueError:
        pass
    wp_ver = Path("/Applications/WirePod.app/Contents/Resources/version")
    return {
        "managed": True,
        "auto": read_env(env_path()).get("CLOAK_AUTO_UPDATE", "1") != "0",
        "installed": text("app/.sha"),
        "channel": text("channel") or "main",
        "status": status,
        "wirepod": {
            "managed": bool(text("wirepod.managed")),
            "installed": wp_ver.read_text().strip() if wp_ver.exists() else "",
            "pinned": ((compat.get("wirepod") or {}).get("macos") or {}).get("pinned", ""),
        },
    }


async def probe_ollama(http: httpx.AsyncClient) -> str | None:
    """A local Ollama with a model pulled? Return the model to use."""
    try:
        models = [m["name"] for m in (await http.get("http://localhost:11434/api/tags")).json().get("models", [])]
    except Exception:
        return None
    for want in ("llama3.2", "llama3.1", "qwen2.5", "gemma", "mistral", "phi"):
        hit = next((name for name in models if name.startswith(want)), None)
        if hit:
            return hit
    return models[0] if models else None


async def autoconfigure(brain: "Brain") -> dict[str, Any]:
    """Configure everything we can find on our own. Only stops to ask when it has to.

    states: done | no_pod | no_bots (pair Vector in wire-pod) | pick (several robots)
    """
    env = read_env(env_path())
    if env.get("CLOAK_SETUP_DONE") == "1":
        return {"state": "done"}
    cfg = brain.cfg
    known = (cfg.get("wirepod", {}) or {}).get("url")
    async with httpx.AsyncClient(timeout=2.5) as http:
        found = await asyncio.gather(*(probe_pod(u, http) for u in dict.fromkeys([known, *POD_GUESSES]) if u))
        pod = next((f for f in found if f and f["bots"]), None) or next((f for f in found if f), None)
        if pod is None:
            return {"state": "no_pod"}
        ready = [b for b in pod["bots"] if b["ready"]]
        if not ready:
            return {"state": "no_bots", "pod": pod["url"]}
        wanted = (cfg.get("robot", {}) or {}).get("serial")
        bot = next((b for b in ready if wanted and b["serial"].lower() == str(wanted).lower()), None)
        if bot is None and len(ready) > 1:
            return {"state": "pick", "pod": pod["url"], "bots": ready}
        bot = bot or ready[0]
        model = None      # Cloak's API is the default LLM now; a local one is opt-in in Settings / .env

    updates = {"WIREPOD_URL": pod["url"], "VECTOR_SERIAL": bot["serial"], "CLOAK_SETUP_DONE": "1"}
    cfg.set_path("wirepod.url", pod["url"])
    cfg.set_path("robot.serial", bot["serial"])        # the link picks these up on its next connect attempt
    if model:
        updates.update(LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL=model)
        cfg.set_path("llm.base_url", updates["LLM_BASE_URL"])
        cfg.set_path("llm.model", model)
        from ..voice.llm import LineWriter
        old, brain.voice.llm = brain.voice.llm, LineWriter(cfg)
        await old.aclose()
    write_env(env_path(), updates)
    brain.bus.emit("setup.auto", pod=pod["url"], serial=bot["serial"], llm=model)
    return {"state": "done", "pod": pod["url"], "serial": bot["serial"], "llm": model}


async def autoconfigure_loop(brain: "Brain") -> None:
    """Keep trying in the background until the robot is found (e.g. while you pair it in wire-pod)."""
    while True:
        try:
            if (await autoconfigure(brain))["state"] == "done":
                return
        except Exception:
            pass
        await asyncio.sleep(5)


EP_OTA = "/api/get_ota/vicos-2.0.1.6076ep.ota"   # wire-pod serves the escape-pod firmware here


def lan_ip() -> str:
    """This machine's address on the network (what the robot can reach), not 127.0.0.1."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))          # no packet is sent; just picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def robot_ota_url(pod_url: str) -> str:
    """The firmware URL as Vector must see it: wire-pod's LAN address, never localhost."""
    import socket
    from urllib.parse import urlparse
    u = urlparse(pod_url or "http://escapepod.local:8080")
    host = u.hostname or "escapepod.local"
    if host in ("localhost", "127.0.0.1", "::1"):
        host = lan_ip()
    else:
        try:
            host = socket.gethostbyname(host)
        except OSError:
            pass
    return f"http://{host}:{u.port or 8080}{EP_OTA}"


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

    @r.post("/auto")
    async def auto() -> dict[str, Any]:
        return await autoconfigure(brain)

    @r.post("/pod")
    async def set_pod(body: dict = Body(...)) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=3) as http:
            found = await probe_pod(str(body.get("url", "")), http)
        if not found:
            return {"ok": False, "error": f"No wire-pod answered at {body.get('url')}."}
        brain.cfg.set_path("wirepod.url", found["url"])
        write_env(env_path(), {"WIREPOD_URL": found["url"]})
        return {"ok": True, **found}

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

    @r.get("/updates")
    async def updates() -> dict[str, Any]:
        return update_info()

    @r.post("/updates")
    async def set_updates(body: dict = Body(...)) -> dict[str, Any]:
        write_env(env_path(), {"CLOAK_AUTO_UPDATE": "1" if body.get("auto") else "0"})
        return update_info()

    @r.post("/updates/check")
    async def check_now() -> dict[str, Any]:
        home = cloak_home()
        if home is None:
            return {"ok": False, "error": "not an installed copy"}
        # detached: an update restarts this very process
        subprocess.Popen([str(home / "bin" / "cloakctl"), "auto-update", "--now"], start_new_session=True,
                         stdout=open(home / "logs" / "update.log", "a"), stderr=subprocess.STDOUT)
        return {"ok": True}

    return r
