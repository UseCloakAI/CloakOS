# CloakOS: handoff for coding agents

This file is for any coding agent (Claude, Grok, Codex, …) picking up this repo.
It describes the state as of v0.9.3 (October 2026). Read it before changing anything.

## What this is

CloakOS ("Cloak, Vector Edition") is the brain for an **Anki Vector** robot. Vector runs
**wire-pod** (the open-source server that replaces Anki's cloud). All the smarts run on the
user's **Mac**:
- behaviors, the eyes and voice
- a 3D map
- a MuJoCo simulator of Vector
- reinforcement-learning training
- a web UI at `http://127.0.0.1:8765`

Part of the Cloak platform (usecloak.org). Repo: `UseCloakAI/CloakOS`.

## How the user works (important)

- **Answers:** short, plain, coworker tone ("Jarvis style"). No code walkthroughs, just
  outline what changed. The user is not a robotics expert; they want plain words.
- **Shipping:** when a change is done, open a PR, merge it (squash) yourself, and tell
  the user to run `cloakos update`.
- **No long test runs.** The user rejects long-running tests and training runs in the dev
  container. Run quick, targeted tests only (`pytest -k …`, seconds not minutes).
- **Interruptions:** the user often adds requests mid-task. Fold them in; don't drop the
  original work.
- **Uncommitted changes:** a stop hook blocks ending a turn with uncommitted changes. If
  work isn't ready to merge, commit and push it to a feature branch.

## Repo rules

- **Version:** bump it on every PR that ships. `pyproject.toml` `version` and
  `cloakos/__init__.py` `__version__` must match. Exception: the user can ask for no bump.
- **Nous** (the shipped pretrained policies in `cloakos/rl/pretrained/`):
  - When they change, bump `NOUS`/`NOUS_VERSION` in `cloakos/__init__.py`.
  - Then run `python scripts/nous_manifest.py <nous version> <min app version>`.
  - Nous is at **v0.0.0 with no shipped skills**, because the user wiped them.
  - The manifest has an `epoch`. Older downloaded Nous models without it are ignored.
- **Don't** put model names or IDs in commits, PRs or code.
- **Writing style:** match the surrounding code. Comments are short and plain, and explain *why*.

## Running and testing

- User commands: `cloakos open | start | stop | restart | status | logs | awake [hours|off] | update | uninstall`.
  The script is `scripts/cloakctl.sh`, installed to `<home>/bin/cloakctl`.
  - `start` / `stop` also start and quit the macOS WirePod app. Opt out with
    `CLOAK_MANAGE_WIREPOD=0` in `.env`.
  - `awake` keeps the Mac awake with the screen off, until training is done or for N hours.
- Install home: `~/Library/Application Support/CloakOS` (macOS) or `~/.local/share/cloakos`.
  Data lives in `<home>/data` (`policies/`, `episodes/`, `calibration.json`, `arena.json`,
  `keep_training.json`). Settings are in `<home>/.env` (`CLOAK_*`, `WIREPOD_URL`, `CLOAK_DESK`, …).
- Tests use pytest. In a headless Linux container:
  `MUJOCO_GL=osmesa CLOAK_SIM=1 CLOAK_SHARE=0 CLOAK_SETUP_DONE=1 python -m pytest -q tests/test_x.py -k …`.
  - **torch plus osmesa in the same process segfaults.** Run the sb3/torch tests separately
    (`MUJOCO_GL=egl … -k "sb3 or torch or student"`).
  - `test_brain_fake_robot` has a failing face assertion that predates current work.
- **Never** `pkill -f <pattern>` with a pattern that also matches your own shell's command
  line. It kills the shell.
- UI checks: serve `cloakos/web` statically (`python -m http.server` from `cloakos/web`, then open
  `/static/index.html`), mock `/api/*` with Playwright routes, and screenshot.
  - Arena tracking can be tested with Chromium's fake camera
    (`--use-file-for-fake-video-capture=x.y4m`) and synthetic videos made with OpenCV.

## The 3D map is parked (on purpose)

The 3D map (`cloakos/mapping/`, `web/map_api.py`, `web/static/map.js`) is future work: pixel-to-map
AI training that builds a 3D map. It is **disconnected on purpose** since v0.9.5: the brain doesn't
start it, the map router isn't registered, the dashboard has no Map page, voice "go to the mug" is off,
and map calibration is refused. The code stays so it can be wired back in later. Don't reconnect it
unless the user asks.

## Autonomy roadmap

**Goal:** tell Vector "go to that point", "find the mug", "go to the mug you saw earlier", and have it
drive itself, as close to FSD-style autonomy as a desk robot gets.

**Principle: accuracy over speed.** Slower driving and slower inference are fine when they're more
accurate. When data is stale or missing, stop and wait rather than guess (e.g. no ranges = hold still,
not "all clear"). Pick the safer, more accurate side of every tradeoff, including recording quality.

**Teacher / student plan:** the overhead iPhone camera (Arena) is the **teacher**: it sees the robot's
pose, the desk edges and (later) objects, so a driver steering from it acts on near-true ranges.
Vector's own camera **student** (pixels to actions, `rl/student.py`) learns to copy it: wander, avoid
edges, point-to-point. Recordings under the Arena store the teacher signals next to the camera frames
(`arena` and `teacher` in `steps.jsonl`); `student.py --real` learns from teleop steps and from
teacher-driven steps.

**Milestones:**
1. **Arena ranges for the real robot (v0.9.6, done):** `arena/ranges.py`, same fan as the sim's rays
   plus the edge margin. `ranges()` order: 3D map (parked) → sim → Arena → none (logged, shown on the
   Arena page, and map drivers hold still). Today it knows desk edges, robot pose and the charger only.
2. An overhead object detector (YOLO) trained on Arena frames, so objects become obstacles in the
   Arena ranges and get labels.
3. **Go-to-point in Arena desk coordinates (v0.9.8, done):** Arena page "Go to a spot", `POST /api/arena/goto`
   (`{x, y}` metres or `{name: "back left corner"}`), or say "go to the back left corner" / "go to 30, 20" (cm).
   `GoToPolicy` in the "arena" frame: the trained goto skill if one ships, else a slow controller
   (70 mm/s max, 25 mm/s inside 12 cm, arrives within 2.5 cm); stops for an edge ahead; holds still when
   the overhead pose is stale. Each tick records `teacher` = goal, `goal_rel` (`envs.goal_features`, the
   same goal input sim goto training uses) and the action.
3b. **Goal-conditioned camera student (v0.9.16, code ready, not trained yet):** `rl/goal_student.py`
   (Training skill "goto-pixels"). Inputs: camera (this + previous frame, lagged/dropped like the real
   stream), body sensors (wheels, distance sensor, cliff, upright, last action) and `goal_features` from
   Vector's OWN dead-reckoned odometry since the start (never the overhead pose). Teacher: a trained
   goto/map driver, else a privileged point-and-drive controller on the true sim pose and exact ranges
   (forward only). DAgger rounds as in `student.py`. Real data: Arena go-to recordings, the goal is moved
   into the odometry frame once at the start of each go-to (`real_goal_steps`). Loading it as a driver is
   refused until it passes the sim exam and gets wired into `LearnedPolicy` (next).
4. Find an object (search until the detector sees it).
5. Remembered objects without the 3D map: an Arena/world object memory (label, desk x/y, last seen).
6. Camera-student distillation for each skill, so it works without the overhead camera.
7. A natural-language command layer on top (voice/LLM picks the skill and target).

**Latency (always there over Wi-Fi):** measured Oct 28: command to motion 0.41 s, stop 0.446 s,
wheel speed ratio 0.86, camera 5 fps, command to picture 0.522 s, command to head sensor 0.268 s,
picture 0.29 s behind the sensors. The robot acts on information about 0.5–0.9 s old. So Arena ranges
(v0.9.7) predict the pose forward by the pose's age + 0.41 s, and shorten the rays ahead by a
stopping buffer (speed × (0.41 + 0.446) s + 2 cm). Training must inject these delays too, with domain
randomization around the measured values (envs.py already does 3–5 steps of action lag, wheel gain
0.82–0.92 and the camera lag in `LaggyCamera`; keep it that way and add Arena pose lag when teacher
drivers are trained on Arena-style ranges), so students learn with the real delay.

The 3D map stays parked (see above).

## Architecture map (`cloakos/`)

- **`app.py`:** `Brain`. The 10 Hz tick loop, the modes
  (`auto | teleop | idle | learned | goto | home`), and the rules around the robot:
  - **charging rules:** never fight Vector's own docking; low battery sends it home.
  - **freeze:** stops the wheels and lift only.
  - **emergency stop:** Shift+Space or the button; halts everything except the face and sound.
  - **Keep training tick:** every 30 s.
- **`connection/`:**
  - `robot.py` (`RobotLink`: connecting, control priority, the **edge reflex**, the volume sync with wire-pod)
  - `body.py` (`Body`: the motor commands; `FROZEN_BLOCKED`, `HALT_OK`)
- **`vector/`:** the gRPC client for Vector's SDK, discovery through wire-pod, the protobufs.
  The robot runs at **override** control priority, so Vector's own firmware edge stop is OFF.
  The brain's edge reflex is the only safety net.
- **`sim/`:**
  - `engine.py`: the MuJoCo `Sim`. The physics stays as it is: tread contact is sensitive,
    and cheaper solver settings changed tread motion by 2–25 cm.
  - `rooms.py`: the generated worlds, including the "desk" world.
  - `deskrun.py`, `tester.py`.
- **`rl/`** (training):
  - `envs.py`:
    - `VectorEnv` with tasks `goto | wander | desk | desk-pixels | recover`.
    - `EdgeReflex`: a model of the real reflex.
    - Wi-Fi lag: an action delay of 3–5 steps at 10 Hz, and wheel gain 0.82–0.92 (both measured).
    - `last_action` is the *commanded* action.
    - `DESK_REWARD_V = 2`.
  - `train.py`: PPO (SB3).
    - Linear learning-rate decay, `target_kl=0.03`.
    - Keeps the best checkpoint in `best/`.
    - `--hours` sets a time budget instead of a step count.
    - Checkpoints every 60 s, written atomically.
    - At the end, `evaluate --pick` tests the best and the last checkpoint and keeps the better one.
  - `student.py`: **DAgger**. The pixel "Camera driver" copies the desk teacher.
    - Sim camera frames go through `LaggyCamera`, using the measured camera lag.
    - `--real` adds teleop recordings, weighted 3x.
    - `--init` stacks onto an earlier student; `--hours` sets a time budget.
  - `pixelnet.py`: numpy inference for the student.
  - `policy.py` (`LearnedPolicy`, `robot_snapshot`, `ranges`) and `skills.py`: run policies on the robot.
  - `evaluate.py`: the exam. 20 unseen worlds (seeds 900001+), run in parallel.
    - Records `hard` seeds; Build on oversamples them 25% of the time (the "data engine").
    - Compares against the parent run.
  - `manager.py`: `Trainer`.
    - Launches subprocesses (under `caffeinate` on macOS).
    - `plan()`: the guided 3-step path.
    - `train_more()`.
    - **Keep training:** back-to-back 2 h sessions, an auto-resume after a crash, a history log.
- **`arena/session.py`, `web/static/arena.js`:** the real desk, watched by an iPhone camera
  above it (Continuity Camera).
  - Tracking runs in the browser:
    - **front/back colour markers**: yellow at the front, red at the back on the user's robot
    - background-difference "shape" fallback
    - a motion filter with gating and coasting
    - the gyro fused in from the backend
  - Scored real runs.
  - The charger location, with routing to it and then a handover to Vector's own docking.
  - Teleop mode while the page is open.
  - `tracker.py`: an unused ArUco tracker, for a possible printed-tag upgrade.
- **`web/`:** FastAPI routers (`server.py`, `rl_api.py`, `sim_api.py`, `arena_api.py`,
  `setup_api.py`, `map_api.py`, `nous_api.py`). The static UI is in `web/static/`:
  - `index.html`
  - `app.js` (the shell, teleop keys with a stuck-key timeout)
  - `train.js`, `sim.js`, `arena.js`, `map.js`
  - `vector.css` (brand tokens: `--acc`, `--ink`, `--paper`, `--bd`, `--shsm`, `--shlg`; hard
    borders, square corners)
- **Other modules:**
  - `mapping/`: the 3D map
  - `expression/`: eyes, sfx, emotes
  - `behaviors/`: Auto mode rules
  - `voice/`
  - `datalog/`: recordings
  - `nous.py`: pretrained skill updates
  - `share.py`: uploads runs and calibration to Supabase project `ufhzuntiekxtsipkhmwn`;
    the publishable key is insert-only.

## Training: the user's view (keep it this simple)

The Training page:
- **3 steps:**
  1. Simulator: the desk teacher, which sees exact edges; sim-only.
  2. Camera: the pixel student, made in the sim; no robot needed.
  3. Your desk: the real teleop recordings improve the camera driver.
- **One choice:** 1 hour, Overnight (8 h), or Keep going.
- **One Start button.** The server picks the next step and what to continue from.
- **Names in the UI:**
  - policy → "driver"
  - teacher → "Simulator driver"
  - student → "Camera driver"
  - exam → "test"
  - build on → "continue from"
- **Advanced:** everything technical goes in the collapsed section.
- Measured real numbers, in `data/calibration.json` on the user's Mac:
  - command to motion 0.41 s, wheel gain 0.86
  - camera 5 fps, the picture 0.29 s behind the sensors
  - The distance sensor reads 0. This is unresolved.

## Current state and what's next

- **v0.9.3:** desk practice v2 is in place:
  - 1–6 objects
  - a cost for getting near an object
  - a roaming reward instead of the old one that allowed loops
  - **Train more** on every driver card
  - the Keep going log
- **The user's simulator driver** `desk-map-20261029-001833`:
  - Passed the v1 test: 100% stays on the desk, 0.15 bumps/min.
  - It's great at edges, weak at avoiding objects, and loops the desk.
  - Under v2 rules the plan sends it back to Step 1 to continue training. The user is
    running Keep going overnight with `cloakos awake`.
- **Keep going stopped after one session once.** The cause is unknown; the history log was
  added to diagnose it. On the next report, check "Keep going: what happened" or
  `data/keep_training.json`.
- **Possible next work** (ask the user first):
  - A printed AprilTag/ArUco marker for the Arena (`arena/tracker.py` exists).
  - A YOLO detector, trained on frames the Arena labels automatically.
  - Renaming the rest of the UI ("Drive lab", …).
  - Making the trained skills ship as Nous.
  - Fixing the distance sensor reading 0, and `test_brain_fake_robot`.
- **Hardware notes:**
  - Vector needs 2.4 GHz Wi-Fi.
  - On an iPhone hotspot, turn on "Maximize Compatibility".
  - macOS Local Network permission is required: "no route to host" means it's missing.
  - The Mac firewall must allow wire-pod (`chipper`).

- `cloakos/web/static/arena.js` has `JS_V`: bump it with the version so a stale Arena tab tells you to reload.
