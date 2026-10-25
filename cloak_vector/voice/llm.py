"""LineWriter: asks an OpenAI-compatible endpoint (Cloak, Ollama, LM Studio, ...) for a tiny line."""
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
        self.enabled = bool(c.get("enabled", True)) and bool(c.get("base_url"))
        self.base_url = (c.get("base_url") or "").rstrip("/")
        self.api_key = c.get("api_key") or ""
        self.model = c.get("model", "")
        self.timeout = float(c.get("timeout_s", 2.5))
        self.temperature = float(c.get("temperature", 1.0))
        self.max_words = int(c.get("max_words", 6))
        self.persona = c.get("persona") or DEFAULT_PERSONA
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._http = httpx.AsyncClient(timeout=self.timeout, headers=headers)

    async def chat(self, user: str, max_tokens: int = 20) -> str | None:
        if not self.enabled:
            return None
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.persona}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        try:
            r = await self._http.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            log.debug("llm: %s", e)
            return None
        return clean_line(text, self.max_words) or None

    async def line(self, situation: str, drives: dict[str, float]) -> str | None:
        mood = ", ".join(f"{k} {v:.1f}" for k, v in drives.items())
        return await self.chat(f"Your feelings (0-1): {mood}.\nWhat just happened: {situation}\nSay one short line.")

    async def reply(self, heard: str, drives: dict[str, float]) -> str | None:
        mood = ", ".join(f"{k} {v:.1f}" for k, v in drives.items())
        return await self.chat(f"Your feelings (0-1): {mood}.\nYour human says: {heard}\nAnswer in one short line.")

    async def aclose(self) -> None:
        await self._http.aclose()
