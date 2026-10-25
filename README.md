# Cloak for Vector

A living-creature brain for an OEM Anki/DDL Vector, part of [Cloak](https://usecloak.org).
**The PC is the brain, Vector is the body.** No firmware mods: Vector talks to your PC over Wi-Fi through [wire-pod](https://github.com/kercre123/wire-pod) and the SDK.

```
 Vector ──Wi-Fi/gRPC──▶ RobotLink ──▶ World ──▶ Senses ──▶ bus ──▶ Drives ──▶ Policy ──▶ Body ──▶ Vector
                                                     │                 │                     ▲
                                                     └──▶ Voice (LLM → Kokoro/Piper → 16 kHz WAV)
                                                     └──▶ Recorder (JSONL + JPEG episodes)
                                    Web dashboard (Cloak chat UI) ◀──WebSocket──▶ modes · teleop · say
```

## Layout

```
cloak_vector/
  __main__.py      python -m cloak_vector
  app.py           Brain: wiring, 10 Hz tick loop, modes (auto / teleop / idle)
  config.py        config.yaml + .env overrides
  bus.py           asyncio pub/sub (sense.*, drive.*, voice.said, behavior.changed, ...)
  world.py         latest sensor snapshot
  connection/      robot.py  connect · heartbeat · reconnect w/ backoff · clean shutdown
                   body.py   the one gateway to motors/anims/speaker; defines the Action space
  perception/      senses.py raw state → debounced sense events   (YOLO slots in here, phase 2)
                   ears.py   optional loud-sound detector on the PC mic
  drives/          model.py  pure drive math, unit-tested
  behaviors/       base.py   Behavior + Policy interfaces   (learned policy slots in here, phase 2)
                   library.py wander · look_around · approach_face · sulk · nap · startle · idle
                   rules.py  RulePolicy: score, pick, run, anti-flicker
  voice/           llm.py (OpenAI-compatible) · tts/ (kokoro, piper) · audio.py (16 kHz) · speaker.py (cache, triggers)
  datalog/         recorder.py  episodes for imitation learning
  teleop/          policy.py    keyboard/gamepad axes → Action, dead-man stop
  web/             FastAPI server + static UI (vendored cloak.css from usecloakai/cloak)
config.yaml        all tuning
.env.example       robot + LLM secrets
tests/             no robot needed (includes an end-to-end run against a fake SDK robot)
```

## Setup

### 1. wire-pod
Set up wire-pod on a machine on the same network as Vector ([wire-pod wiki](https://github.com/kercre123/wire-pod/wiki)), then pair the bot through wire-pod's web UI (`http://<pod-ip>:8080`). Once Vector answers "Hey Vector" through wire-pod, you're good.

### 2. Python
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[kokoro]"          # or .[piper]; add ,ears for the PC mic, ,dev for tests
```

### 3. SDK auth (once)
```bash
python -m anki_vector.configure
```
It asks for Vector's serial (on the bottom of the bot, or in wire-pod's bot list), the wire-pod address (`192.168.1.50:8080`, or blank for `escapepod.local`), the robot name (`Vector-A1B2`) and the IP. It pulls the cert from wire-pod and writes `~/.anki_vector/sdk_config.ini`.

### 4. `.env`
```bash
cp .env.example .env
```
Set `VECTOR_SERIAL` (and `VECTOR_IP` if mDNS is flaky). For lines from an LLM, set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` to any OpenAI-compatible endpoint: Cloak, Ollama (`http://localhost:11434/v1`), LM Studio and so on. Without an LLM, Vector speaks from its cached line pool.

### 5. Voice models
Kokoro (the default):
```bash
mkdir -p models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```
Want the full-precision model? Grab `kokoro-v1.0.onnx` and point `voice.tts.kokoro.model` at it. For Piper, set `TTS_ENGINE=piper`, download a voice's `.onnx` and `.onnx.json` from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), and set `voice.tts.piper.model`.

## Run

```bash
python -m cloak_vector                 # auto mode + dashboard at http://127.0.0.1:8765
python -m cloak_vector --mode teleop   # start in teleop (records automatically)
python -m cloak_vector --mode idle     # stock Vector runs; we just watch and log
python -m cloak_vector --no-web -v     # headless, debug logs
```
Switch modes live from the dashboard's mode picker. Ctrl+C stops the motors, saves any recording and disconnects cleanly. If Vector drops off Wi-Fi, the link retries with backoff (1s up to 30s) and picks the mode back up on reconnect.

| Mode | Who drives | Behavior control |
|---|---|---|
| **Auto** | RulePolicy (drives → behaviors) | held by Cloak |
| **Teleop** | you (keyboard / gamepad in the dashboard) | held by Cloak |
| **Idle** | stock Vector personality | released |

## Dashboard
It's styled like Cloak chat: same sidebar, topbar, composer and themes (`cloak_dark` / `cloak_theme` in localStorage).
- **Live**: camera with a face box, drive meters, behavior / battery / proximity / body tiles. Below that, the stream: what Vector says shows up as Cloak messages, and senses, drive crossings and behavior switches show up as event rows.
- **Composer**: **Say it** makes Vector speak your text as written. **Talk to it** sends your text to the LLM, and Vector answers in character.
- **Teleop**: `W A S D` to drive, `↑ ↓` head, `R F` lift, `Shift` boost, `Space` stop. Gamepads work too (left stick drives, right stick moves the head, triggers move the lift, A is boost). Input stops if the tab goes quiet for 0.5s (dead-man).
- **Drive lab**: fire fake senses, drag drive values around, and watch the live behavior scores. Handy for tuning without the bot.
- **Settings**: connection, voice and LLM status, and every animation trigger your Vector has, with a test player.

Keep `cloak_vector/web/static/cloak.css` an exact copy of `usecloakai/cloak/cloak.css`. Put overrides in `vector.css`, and to sync, copy the file over again.

## Drive system

Five drives, each a float in `[0, 1]`:

| Drive | Baseline / τ | Rises with | Falls with |
|---|---|---|---|
| energy | 0.6 / 30 min | napping | moving, low battery (capped at 0.3) |
| curiosity | 0.35 / 2 min | idle time, faces, sounds, wake word | cliff |
| boredom | 0 / ~7 min | idle time | faces, touch, wake word |
| attachment | 0.2 / 5 min | faces, petting | time |
| fear | 0 / 8 s | pickup, cliff, loud sounds | touch, faces, time (fast) |

Every tick: `v += (baseline − v)·(1 − e^(−dt/τ)) + drift·dt`, clamped. Sense events add the deltas from the `impulses` table. When a drive crosses a `high` or `low` line it fires `drive.<name>.high|low` once, then re-arms after dropping back past the `hysteresis` band. All of it is set in `config.yaml`.

## Behaviors

| Behavior | Score (× weight) | What it does |
|---|---|---|
| startle | `1.6·fear` if fear > 0.6, **reflex** | scared anim, backs off |
| approach_face | `0.55 + 0.6·attachment − 0.4·fear` while a face is visible | centers the face, drives up, greets |
| nap | `1.3·(1−energy)` when energy < 0.35 | sleep anims until energy > 0.85 (energy recovers) |
| sulk | `1.2·boredom·(1−attachment)` | head down, sad anim, turns away |
| wander | `0.45·curiosity + 0.45·energy − 0.25·boredom` | random-walk drive, avoids cliffs and obstacles |
| look_around | `0.15 + 0.5·curiosity + 0.2·(1−energy)` | head and body scans |
| idle | `0.2` | sits and glances around |

The highest score wins. A running behavior keeps a +0.15 stickiness bonus and its `min_s` before it can be replaced, but reflexes cut in right away. Animations are picked by keyword from Vector's own trigger list, which blends stock animations into the rule behaviors.

## Voice
A trigger (a drive crossing like `fear.high`, or a sense like `picked_up`) leads to a line, then TTS, then a 16 kHz mono 16-bit WAV, then Vector's speaker.
- Every fallback line in `voice.triggers` is rendered at startup, so those play with no TTS wait.
- LLM lines race `llm.timeout_s` (2.5s). A slow or missing LLM falls back to a cached line.
- Every rendered line is cached on disk by its text (LRU, 500 files).
- There's a global cooldown, a per-trigger cooldown and a `chance` setting, and speech never queues up.
- `voice.pitch` (default 1.15) resamples the audio for a higher, quicker creature voice.

Kokoro int8 takes roughly 0.5s per short line on a plain CPU. Cached lines skip that.

## Data format (imitation learning)
`data/episodes/<UTC>_<mode>/`:
- `meta.json`: fps, frame size, action space, and a config snapshot with secrets stripped
- `steps.jsonl`: one line per tick
  ```json
  {"i": 42, "t": 1729830000.1, "mode": "teleop", "obs": {"head_angle": 0.17, "lift_mm": 32.0, "left_mmps": 148.2, "proximity_mm": 210.0, "picked_up": false, "face": {"cx": 0.52, "size": 0.14}, "...": "..."},
   "drives": {"energy": 0.71, "...": "..."}, "behavior": "teleop",
   "action": {"left_mmps": 150, "right_mmps": 150, "head_rate": 0, "lift_rate": 0}, "action_source": "teleop",
   "events": [{"topic": "sense.touched"}, {"kind": "anim", "trigger": "GreetAfterLongTime"}], "frame": "frames/0000042.jpg"}
  ```
- `frames/*.jpg`: 320×180 camera frames

`action` is the continuous action space, the same one teleop produces and a learned policy will output. High-level moves (turns, anims, speech) are logged as discrete `events`. Each recording is one episode. Teleop records automatically, and the record button covers everything else.

## Phase 2 hooks
- **Vision (YOLO)**: a detector in `perception/` publishes `raw.face` / `raw.object` events on the bus. Senses, drives and behaviors don't change.
- **Learned policy (ACT / diffusion)**: subclass `behaviors.base.Policy`, read the `Obs`, and send `body.apply(Action(...))` each step. Then register it in `Brain.policies` as a new mode. Write an exporter that turns `data/episodes` into LeRobot or HDF5.
- **Personality LLM**: it's just `llm.base_url` / `llm.model` / `llm.persona`.

## Tests
```bash
pip install -e ".[dev]" && pytest
```
The tests cover drive math, behavior selection, teleop mapping, the WAV format, the recorder, and a full brain run against a fake SDK robot (connect, auto, teleop with recording, idle, shutdown).

## Notes
- The SDK doesn't expose Vector's microphone, so loud sounds come from the PC mic (`senses.ears.enabled: true`, `pip install -e ".[ears]"`). The "Hey Vector" wake word comes through the SDK as a sense event.
- In Auto and Teleop, Cloak holds behavior control at default priority. Vector's own safety reactions, like the tuck when it's falling, still win.
