"""LineWriter: asks an LLM for a tiny line or a decision.

Default provider is Cloak (api.usecloak.org, public guest access, no key): `pneuma` for the
quick 1-6 word lines, `logos` for understanding what you asked it to do. Any OpenAI-compatible
endpoint (Ollama, LM Studio, ...) works too with provider: openai.
"""
from __future__ import annotations

import logging
import re

import httpx

log = logging.getLogger(__name__)

DEFAULT_PERSONA = (
    "You are a small, curious robot creature living on a desk. You are not an assistant. "
    "You speak in 1 to 6 words, simple and warm, like a pet that just learned to talk. "
    "No emoji, no quotes, no narration, no explanations. Just the words you say out loud."
)


def clean_line(text: str, max_words: int = 6) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"\*[^*]*\*|\([^)]*\)", "", text)              # *beeps*, (softly)
    text = re.sub(r"^[\"'*\s]+|[\"'*\s]+$", "", text)
    text = re.sub(r"[^\w\s,.!?'-]", "", text)                     # emoji, odd symbols
    words = text.split()
    return " ".join(words[:max_words]).strip(" ,")


class LineWriter:
    def __init__(self, cfg: dict) -> None:
        c = cfg.get("llm", {}) or {}
        url = c.get("base_url") or ""
        # auto: Cloak, unless you pointed base_url at something else (e.g. Ollama from the setup wizard)
        self.provider = (c.get("provider") or ("openai" if url and "usecloak" not in url else "cloak")).lower()
        if self.provider == "cloak":
            self.base_url = (c.get("base_url") or "https://api.usecloak.org").rstrip("/")
            self.model = c.get("model") or "pneuma"
            self.chat_model = c.get("chat_model") or "logos"
        else:
            self.base_url = (c.get("base_url") or "").rstrip("/")
            self.model = c.get("model", "")
            self.chat_model = c.get("chat_model") or self.model
        self.enabled = bool(c.get("enabled", True)) and bool(self.base_url)
        self.api_key = c.get("api_key") or ""
        self.timeout = float(c.get("timeout_s", 2.5))
        self.temperature = float(c.get("temperature", 1.0))
        self.max_words = int(c.get("max_words", 6))
        self.persona = c.get("persona") or DEFAULT_PERSONA
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._http = httpx.AsyncClient(timeout=self.timeout, headers=headers)

    async def ask(self, user: str, system: str, model: str | None = None, max_tokens: int = 200,
                  temperature: float | None = None, timeout: float | None = None,
                  image_jpeg: bytes | None = None) -> str | None:
        """Raw text from the model, or None (offline, slow, error)."""
        if not self.enabled:
            return None
        temp = self.temperature if temperature is None else temperature
        try:
            if self.provider == "cloak":
                body = {"model": model or self.model, "message": user, "system_prompt": system,
                        "temperature": temp, "effort": "low"}
                if image_jpeg:
                    import base64
                    body["imageBase64"], body["mimeType"] = base64.b64encode(image_jpeg).decode(), "image/jpeg"
                r = await self._http.post(f"{self.base_url}/v1/chat", json=body, timeout=timeout or self.timeout)
                r.raise_for_status()
                return (r.json().get("text") or "").strip() or None
            body = {"model": model or self.model, "max_tokens": max_tokens, "temperature": temp,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            r = await self._http.post(f"{self.base_url}/chat/completions", json=body, timeout=timeout or self.timeout)
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip() or None
        except Exception as e:
            log.debug("llm: %s", e)
            return None

    async def chat(self, user: str, max_tokens: int = 20) -> str | None:
        text = await self.ask(user, self.persona, max_tokens=max_tokens)
        return clean_line(text, self.max_words) if text else None

    async def line(self, situation: str, drives: dict[str, float]) -> str | None:
        mood = ", ".join(f"{k} {v:.1f}" for k, v in drives.items())
        return await self.chat(f"Your feelings (0-1): {mood}.\nWhat just happened: {situation}\nSay one short line.")

    async def reply(self, heard: str, drives: dict[str, float]) -> str | None:
        mood = ", ".join(f"{k} {v:.1f}" for k, v in drives.items())
        return await self.chat(f"Your feelings (0-1): {mood}.\nYour human says: {heard}\nAnswer in one short line.")

    async def aclose(self) -> None:
        await self._http.aclose()
