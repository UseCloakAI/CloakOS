"""Commander: what you say to Vector -> what it says back + what it does.

The LLM (Cloak's `logos` by default) sees the objects Vector remembers and answers with a
tiny JSON decision: a line to say and an action (go to something, wander, stop). If the LLM
is offline, plain phrases still work: "go to the mug", "find the shoe", "stop", "wander".
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

ACTIONS = ("go_to", "wander", "stop", "none")

SYSTEM = """You are the mind of Vector, a small curious desk robot. Your human talks to you.
Decide what to say and what to do. You can only do these actions:
- "go_to": drive to one of the objects you remember (use its id from the list)
- "wander": roam around exploring
- "stop": stop moving and stay put
- "none": just talk

Objects you remember (id, what it is, how far from you):
{objects}

Reply with ONLY a JSON object, nothing else:
{{"say": "<1 to 8 words, warm and simple, no emoji>", "action": "go_to|wander|stop|none", "target": "<object id or null>"}}
If they ask you to go to something you don't remember, say you haven't seen it and use "none"."""


@dataclass
class Decision:
    say: str | None
    action: str
    target: str | None = None
    source: str = "llm"


def _objects_text(objects: list[dict], here: tuple[float, float] | None) -> str:
    if not objects:
        return "(nothing yet)"
    import math

    rows = []
    for o in objects[:30]:
        d = f"{math.dist(here, o['pos'][:2]):.1f} m" if here else "?"
        rows.append(f"- {o['id']}: {o['label']}, {d}")
    return "\n".join(rows)


def parse(text: str, objects: list[dict]) -> Decision | None:
    """Pull the JSON decision out of a model reply (tolerates extra words around it)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    action = str(d.get("action") or "none").strip().lower()
    if action not in ACTIONS:
        action = "none"
    target = d.get("target")
    if action == "go_to":
        ids = {o["id"] for o in objects}
        if target not in ids:                   # the model used a word instead of an id
            want = str(target or "").lower()
            hit = next((o for o in objects if o["label"].lower() in want or (want and want in o["label"].lower())), None)
            target = hit["id"] if hit else None
            if target is None:
                action = "none"
    say = str(d.get("say") or "").strip() or None
    return Decision(say, action, target)


GO_RE = re.compile(r"\b(?:go|drive|head|move|come)\s+(?:to|over to|toward|towards)\s+(?:the|my|a|that)?\s*(.+)|\bfind\s+(?:the|my|a)?\s*(.+)", re.I)


def fallback(text: str, objects: list[dict]) -> Decision:
    """No LLM: understand the obvious phrasings."""
    t = text.strip().lower().rstrip(".!?")
    if re.search(r"\b(stop|stay|freeze|halt|wait)\b", t):
        return Decision("Okay, stopping.", "stop", source="rules")
    if re.search(r"\b(wander|explore|roam|go play|look around)\b", t):
        return Decision("Off I go!", "wander", source="rules")
    m = GO_RE.search(t)
    if m:
        what = (m.group(1) or m.group(2) or "").strip()
        words = [w for w in re.findall(r"[a-z]+", what) if w not in ("the", "my", "a", "please", "now")]
        for w in words or [what]:
            hit = next((o for o in objects if w in o["label"].lower()), None)
            if hit:
                return Decision(f"Going to the {hit['label']}!", "go_to", hit["id"], source="rules")
        return Decision(f"I haven't seen a {' '.join(words) or what} yet.", "none", source="rules")
    return Decision(None, "none", source="rules")


async def decide(llm, text: str, objects: list[dict], here: tuple[float, float] | None) -> Decision:
    quick = fallback(text, objects)
    if quick.action != "none":                  # clear commands act instantly, no round trip
        return quick
    reply = await llm.ask(text, SYSTEM.format(objects=_objects_text(objects, here)),
                          model=getattr(llm, "chat_model", None), temperature=0.4, timeout=6.0)
    d = parse(reply, objects) if reply else None
    if d is None:
        if quick.say:
            return quick
        if reply:                               # it answered, just not in JSON: treat it as talk
            return Decision(reply.strip().splitlines()[0][:120], "none", source="llm")
        return quick
    return d
