"""Config: config.yaml for tuning, .env for secrets and per-machine values.

Env vars override YAML so the same config.yaml works on any PC.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# env var -> dotted config path
ENV_OVERRIDES = {
    "VECTOR_SERIAL": "robot.serial",
    "VECTOR_IP": "robot.ip",
    "VECTOR_NAME": "robot.name",
    "LLM_BASE_URL": "llm.base_url",
    "LLM_API_KEY": "llm.api_key",
    "LLM_MODEL": "llm.model",
    "TTS_ENGINE": "voice.tts.engine",
    "WEB_HOST": "web.host",
    "WEB_PORT": "web.port",
    "DATA_DIR": "datalog.dir",
}


class Config(dict):
    """dict with dotted get: cfg.get_path("drives.fear.tau")."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set_path(self, path: str, value: Any) -> None:
        node = self
        *parents, last = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[last] = value


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    load_dotenv()
    path = Path(path)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = Config(data or {})
    for env, dotted in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg.set_path(dotted, _coerce(os.environ[env]))
    return cfg
