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

ACTIONS = ("go_to", "wander", "stop", "go_home", "moves", "none")
MOVES = ("forward", "back", "turn", "head", "lift", "wait")
MAX_MM, MAX_STEPS = 500, 10

SYSTEM = """You are the mind of Vector, a small curious desk robot. Your human talks to you.
Decide what to say and what to do. You can only do these actions:
- "go_to": drive to one of the objects you remember (use its id from the list)
- "wander": roam around exploring
- "stop": stop moving and stay put
- "go_home": drive back onto the charger
- "moves": precise steps, done in order, e.g. [["back", 50], ["turn", 90], ["forward", 100]]:
    forward/back <mm, up to 500>, turn <degrees, + = left, - = right>, head <degrees -20..45>,
    lift <0 = down .. 1 = up>, wait <seconds>
- "none": just talk

Objects you remember (id, what it is, how far from you):
{objects}

Reply with ONLY a JSON object, nothing else:
{{"say": "<1 to 8 words, warm and simple, no emoji>", "action": "go_to|wander|stop|go_home|moves|none", "target": "<object id or null>", "steps": [["forward", 100]]}}
If they ask you to go to something you don't remember, say you haven't seen it and use "none"."""


@dataclass
class Decision:
    say: str | None
    action: str
    target: str | None = None
    source: str = "llm"
    steps: list | None = None


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
    steps = clean_steps(d.get("steps")) if action == "moves" else None
    if action == "moves" and not steps:
        action = "none"
    return Decision(say, action, target, steps=steps)


def clean_steps(raw) -> list | None:
    """Only known moves, numbers clamped to safe ranges, at most MAX_STEPS."""
    out = []
    for st in (raw or [])[:MAX_STEPS]:
        try:
            kind, val = str(st[0]).lower(), float(st[1])
        except (TypeError, ValueError, IndexError):
            continue
        if kind not in MOVES:
            continue
        lim = {"forward": (0, MAX_MM), "back": (0, MAX_MM), "turn": (-360, 360), "head": (-20, 45),
               "lift": (0, 1), "wait": (0, 5)}[kind]
        out.append([kind, max(lim[0], min(lim[1], val))])
    return out or None


MOVE_RE = re.compile(r"\b(forward|ahead|back(?:\s*up)?|backward|backwards|reverse|turn\s+(?:left|right)|turn\s+around|"
                     r"look\s+(?:up|down)|lift\s+(?:up|down)|raise\s+(?:your\s+)?lift|lower\s+(?:your\s+)?lift)"
                     r"(?:\s+(?:by\s+)?(-?\d+(?:\.\d+)?)\s*(cm|centimeters?|mm|millimeters?|in|inch(?:es)?|deg(?:rees?)?|°)?)?", re.I)


def moves_from_text(t: str) -> list | None:
    """'back up 10 cm then turn left 90' -> [['back', 100], ['turn', 90]]. Missing numbers get gentle defaults."""
    steps = []
    for m in MOVE_RE.finditer(t):
        word, num, unit = m.group(1).lower(), m.group(2), (m.group(3) or "").lower()
        n = float(num) if num else None
        mm = None
        if n is not None:
            mm = n * 10 if unit.startswith("c") or not unit else n if unit.startswith("m") else n * 25.4 if unit.startswith("in") else n
        if word in ("forward", "ahead"):
            steps.append(["forward", mm or 100])
        elif word.startswith("back") or word == "reverse":
            steps.append(["back", mm or 80])
        elif "around" in word:
            steps.append(["turn", 180])
        elif word.startswith("turn"):
            deg = n if n is not None else 90
            steps.append(["turn", deg if "left" in word else -deg])
        elif "look up" in word:
            steps.append(["head", 35])
        elif "look down" in word:
            steps.append(["head", -15])
        elif "up" in word or "raise" in word:
            steps.append(["lift", 1])
        else:
            steps.append(["lift", 0])
    return clean_steps(steps)


GO_RE = re.compile(r"\b(?:go|drive|head|move|come)\s+(?:to|over to|toward|towards)\s+(?:the|my|a|that)?\s*(.+)|\bfind\s+(?:the|my|a)?\s*(.+)", re.I)


def fallback(text: str, objects: list[dict]) -> Decision:
    """No LLM: understand the obvious phrasings."""
    t = text.strip().lower().rstrip(".!?")
    if re.search(r"\b(stop|stay|freeze|halt|wait)\b", t):
        return Decision("Okay, stopping.", "stop", source="rules")
    if re.search(r"\b(go home|go to (?:your )?(?:charger|home|bed)|go charge|charge yourself|dock)\b", t):
        return Decision("Heading home.", "go_home", source="rules")
    steps = moves_from_text(t)
    if steps:
        return Decision(None, "moves", steps=steps, source="rules")
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
