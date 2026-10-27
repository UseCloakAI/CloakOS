# CloakOS

**CloakOS Vector Edition**: a living-creature brain for Anki/DDL Vector, part of [Cloak](https://usecloak.org).

A living-creature brain for an OEM Anki/DDL Vector, part of [Cloak](https://usecloak.org).
**The PC is the brain, Vector is the body.** No firmware mods and no Anki SDK. Everything is custom and built on [wire-pod](https://github.com/kercre123/wire-pod): Cloak asks wire-pod for the robot's address, token and cert, then talks gRPC straight to Vector with its own client.

```
 wire-pod ──HTTP──▶ discovery (serial · IP · token · cert)
 Vector ◀──TLS gRPC──▶ VectorClient ──▶ World ──▶ Senses ──▶ bus ──▶ Drives ──▶ Policy ──▶ Body ──▶ Vector
                                                         │               │                    ▲
                                                         ├──▶ Expression (Cloak eyes · chirps · emotes)
                                                         ├──▶ Voice (LLM → Kokoro/Piper → 16 kHz PCM)
                                                         └──▶ Recorder (JSONL + JPEG episodes)
                                    Web dashboard (Cloak chat UI) ◀──WebSocket──▶ modes · teleop · say
```

## Install (macOS)

One command:

```bash
curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakVOS/main/install.sh | bash
```

That's the only command. It installs into `~/Library/Application Support/CloakOS`:
- Python (via [uv](https://docs.astral.sh/uv/), no Homebrew needed), the app and the voice model.
- **wire-pod**, if there isn't one on your network yet. That's the official [WirePod](https://github.com/kercre123/WirePod) app, at the version pinned in `compat.json`. Its first-run questions are answered for you: it starts at login, and its speech language is picked from your Mac's language.
- **CloakOS** in `~/Applications`, plus a `cloakos` command.
- Background jobs: Cloak starts when you log in, and checks for updates every 6 hours.

Then it opens Cloak in your browser and configures itself. It finds wire-pod and your robot, and uses a local LLM if Ollama is running. The only thing it can't do for you is introduce Vector to wire-pod: that's a Bluetooth pairing in wire-pod's **Bot Setup** page. If that hasn't happened yet, Cloak shows the 3 steps and carries on by itself the moment Vector shows up. Voice and brain are optional ("Customize voice & brain").

What you might still see on a Mac: macOS asking once to allow "local network" access (click Allow), and, if your user isn't an admin, a password prompt to put WirePod in `/Applications`.

### Uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakVOS/main/uninstall.sh | bash
```

Or `cloakos uninstall`. It asks once, then removes everything CloakOS installed:
- the app, the `cloakos` command, start-at-login and auto-update
- the install folder (Python environment, voice model, settings)
- **WirePod**: the app, its login item and its settings, if CloakOS installed it
- **uv** and the Python it downloaded, if CloakOS installed them

A wire-pod or uv you had before CloakOS is left alone. Your recordings are moved to `~/Documents/CloakOS Recordings` rather than deleted; pass `--delete-recordings` to delete them. Pass `--yes` to skip the question.

### Updates
- **CloakOS** updates itself in the background (every 6 hours) from `main`. It never interrupts a recording, and you can turn it off in Settings → Updates or run `cloakos update` any time.
- **wire-pod** only updates to a version we've deemed compatible. Every Monday, the `wire-pod compatibility` workflow checks the newest WirePod release against everything Cloak uses (`scripts/wirepod_contract.py`). If it passes, the workflow opens a PR bumping `compat.json`. Merging that PR rolls the new wire-pod out to every Mac on its next update. Cloak never touches a wire-pod it didn't install.
- For the workflow to open PRs, enable *Settings → Actions → General → Allow GitHub Actions to create and approve pull requests* in the repo.

Other commands: `cloakos open | stop | status | logs | update | uninstall`.

Upgrading from an older "Cloak Vector" install? It moves to CloakOS by itself on its next update, or when you run the one-liner again. Settings and recordings come along.

To install from a branch: `curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakVOS/<branch>/install.sh | CLOAK_REF=<branch> bash`.

The manual setup below is for development.

## Layout

```
cloakos/
  __main__.py      python -m cloakos
  app.py           Brain: wiring, 10 Hz tick loop, modes (auto / teleop / idle)
  config.py        config.yaml + .env overrides
  bus.py           asyncio pub/sub (sense.*, drive.*, voice.said, behavior.changed, ...)
  world.py         latest sensor snapshot
  vector/          our own robot client (no Anki SDK)
                   discovery.py  wire-pod /api-sdk/get_sdk_info + /session-certs → serial, IP, token, cert
                   client.py     async gRPC: control, events, camera, motors, actions, screen, speaker
                   proto/ + pb/  Vector's external-interface protos (Apache-2.0, Anki) + generated stubs
  connection/      robot.py  discover · connect · heartbeat · reconnect w/ backoff · clean shutdown
                   body.py   the one gateway to motors/screen/speaker; defines the Action space
  expression/      eyes.py   Cloak-styled procedural eyes on Vector's screen, driven by the drives
                   chirps.py synthesized creature sounds
                   emotes.py eyes + motion + chirp clips (greet, happy, curious, scared, sad, ...)
  perception/      senses.py raw state → debounced sense events   (YOLO slots in here, phase 2)
                   ears.py   optional loud-sound detector on the PC mic
  drives/          model.py  pure drive math, unit-tested
  behaviors/       base.py   Behavior + Policy interfaces   (learned policy slots in here, phase 2)
                   library.py wander · look_around · approach_face · sulk · nap · startle · idle
                   rules.py  RulePolicy: score, pick, run, anti-flicker
  voice/           llm.py (OpenAI-compatible) · tts/ (kokoro, piper) · audio.py (16 kHz) · speaker.py (cache, triggers)
  datalog/         recorder.py  episodes for imitation learning
  teleop/          policy.py    keyboard/gamepad axes → Action, dead-man stop
  web/             FastAPI server + static UI (vendored cloak.css from usecloakai/cloak) + first-run setup wizard
config.yaml        all tuning
.env.example       robot + LLM secrets
install.sh         one-command macOS installer
uninstall.sh       removes everything CloakOS installed
compat.json        the wire-pod version Cloak installs/updates to (bumped by the compatibility workflow)
scripts/           cloakctl.sh (cloakos command + updater) · wirepod.sh (install/update WirePod)
                   wirepod_contract.py (compatibility check) · make_icon.py · gen_proto.py
tests/             no robot needed (includes a fake Vector gRPC server over real TLS)
```

## Setup (development)

### 1. wire-pod
Set up wire-pod on a machine on the same network as Vector ([wire-pod wiki](https://github.com/kercre123/wire-pod/wiki)), then pair and authenticate the bot in wire-pod's web UI (`http://<pod-ip>:8080`). Once Vector answers "Hey Vector" through wire-pod, you're good. That's the only robot setup: there's no SDK configure step.

### 2. Python
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[kokoro]"          # or .[piper]; add ,ears for the PC mic, ,dev for tests
```

### 3. `.env`
```bash
cp .env.example .env
```
Set `WIREPOD_URL` (default `http://escapepod.local:8080`). On connect, Cloak reads the bot's serial, IP and token from wire-pod's `/api-sdk/get_sdk_info` and fetches its TLS cert from `/session-certs/<serial>`; the cert is cached in `data/certs/`. Running several bots? Set `VECTOR_SERIAL` to pick one. `VECTOR_IP`, `VECTOR_GUID` and `VECTOR_NAME` override discovery if you ever need to.

For lines from an LLM, set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` to any OpenAI-compatible endpoint: Cloak, Ollama (`http://localhost:11434/v1`), LM Studio and so on. Without an LLM, Vector speaks from its cached line pool.

### 4. Voice models
Kokoro (the default):
```bash
mkdir -p models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```
Want the full-precision model? Grab `kokoro-v1.0.onnx` and point `voice.tts.kokoro.model` at it. For Piper, set `TTS_ENGINE=piper`, download a voice's `.onnx` and `.onnx.json` from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), and set `voice.tts.piper.model`.

## Run

```bash
python -m cloakos                 # auto mode + dashboard at http://127.0.0.1:8765
python -m cloakos --mode teleop   # start in teleop (records automatically)
python -m cloakos --mode idle     # stock Vector runs; we just watch and log
python -m cloakos --no-web -v     # headless, debug logs
```
Switch modes live from the dashboard's mode picker. Ctrl+C stops the motors, hands control back to Vector, saves any recording and disconnects cleanly. If Vector drops off Wi-Fi, or any stream dies, the link re-discovers through wire-pod and reconnects with backoff (1s up to 30s), then picks the mode back up.

| Mode | Who drives | Behavior control |
|---|---|---|
| **Auto** | RulePolicy (drives → behaviors) | held by Cloak |
| **Teleop** | you (keyboard / gamepad in the dashboard) | held by Cloak |
| **Idle** | stock Vector personality | released |

## Dashboard
It's styled like Cloak chat: same sidebar, topbar, composer and themes (`cloak_dark` / `cloak_theme` in localStorage).
- **Live**: camera with a face box and an inset of the eyes Cloak is drawing on Vector's screen (with no camera, the eyes fill the box, so you can watch them offline too), drive meters, behavior / battery / proximity / body tiles. Below that, the stream: what Vector says shows up as Cloak messages, and senses, drive crossings and behavior switches show up as event rows.
- **Composer**: **Say it** makes Vector speak your text as written. **Talk to it** sends your text to the LLM, and Vector answers in character.
- **Teleop**: `W A S D` to drive, `↑ ↓` head, `R F` lift, `Shift` boost, `Space` stop. Gamepads work too (left stick drives, right stick moves the head, triggers move the lift, A is boost). Input stops if the tab goes quiet for 0.5s (dead-man).
- **Drive lab**: fire fake senses, play any emote, drag drive values around, and watch the live behavior scores. Handy for tuning without the bot.
- **Settings**: wire-pod and robot info, voice and LLM status, and every stock animation trigger your Vector has, with a test player.

Keep `cloakos/web/static/cloak.css` an exact copy of `usecloakai/cloak/cloak.css`. Put overrides in `vector.css`, and to sync, copy the file over again.

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

The highest score wins. A running behavior keeps a +0.15 stickiness bonus and its `min_s` before it can be replaced, but reflexes cut in right away. Behaviors express themselves through Cloak's own emotes (below).

## Expression (custom)
- **Eyes**: Cloak's look on Vector's 184×96 screen. Two square rust eyes with a hard offset shadow, redrawn 8×/s. Energy sets how open they are, curiosity their size and asymmetry, attachment gives happy `^ ^` lids, boredom droops them, fear makes them wide and shaky. They follow the face Vector sees, blink, wander when idle and shut during naps.
- **Chirps**: creature sounds synthesized with numpy (gliding notes plus harmonics), with no sound files. They're separate from TTS speech.
- **Emotes**: `greet happy curious scared startle sad bored sleepy wake`. Each one plays eyes, motion (wiggles, nods, head/lift moves) and a chirp together.
- **Reactions**: sense → emote on top of the running behavior (`touched: happy`, `heard_wake_word: curious`, ...).
- **Stock blend**: `expression.stock_blend` (default 0.25) is the chance an emote borrows one of Vector's built-in animations instead. Set it to `0` for a fully custom creature.

## Voice
A trigger (a drive crossing like `fear.high`, or a sense like `picked_up`) leads to a line, then TTS, then 16 kHz mono 16-bit PCM, which our client streams to Vector's speaker in 1 KB chunks paced to real time.
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
   "events": [{"topic": "sense.touched"}, {"kind": "emote", "name": "happy"}], "frame": "frames/0000042.jpg"}
  ```
- `frames/*.jpg`: 320×180 camera frames

`action` is the continuous action space, the same one teleop produces and a learned policy will output. High-level moves (turns, emotes, speech) are logged as discrete `events`. Each recording is one episode. Teleop records automatically, and the record button covers everything else.

## Phase 2 hooks
- **Vision (YOLO)**: a detector in `perception/` publishes `raw.face` / `raw.object` events on the bus. Senses, drives and behaviors don't change.
- **Learned policy (ACT / diffusion)**: subclass `behaviors.base.Policy`, read the `Obs`, and send `body.apply(Action(...))` each step. Then register it in `Brain.policies` as a new mode. Write an exporter that turns `data/episodes` into LeRobot or HDF5.
- **Personality LLM**: it's just `llm.base_url` / `llm.model` / `llm.persona`.
- **Learned expression**: `Eyes.shape()` turns drives into a handful of eye parameters, so a model can drive the same parameters later.

## Tests
```bash
pip install -e ".[dev]" && pytest
```
The tests cover drive math, behavior selection, teleop mapping, audio, the recorder, eyes and chirps, wire-pod discovery (mocked HTTP) and our gRPC client against a fake Vector server over real TLS with token auth. There's also a full brain run against that fake robot: connect, control grant, events, camera, eyes on the screen, auto, teleop with recording, idle, shutdown.

Protos: `cloakos/vector/proto/` holds Anki's Vector external-interface protos (Apache-2.0, via vector-go-sdk) with the REST annotations stripped. The generated stubs are committed. If you edit a proto, run `pip install -e ".[proto]" && python scripts/gen_proto.py`.

## Notes
- Vector's external interface doesn't stream its microphone, so loud sounds come from the PC mic (`senses.ears.enabled: true`, `pip install -e ".[ears]"`). The "Hey Vector" wake word arrives on the event stream as a sense event.
- In Auto and Teleop, Cloak holds behavior control at `robot.control_priority: default`. Vector's own safety reactions, like the tuck when it's falling, still win. `override` disables those too, so be careful.
