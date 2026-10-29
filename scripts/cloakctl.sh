#!/usr/bin/env bash
# cloakos: control the installed CloakOS (Vector Edition).
#   open | start | stop | restart | status | logs | awake [hours|off] | update | auto-update | uninstall
set -euo pipefail

# resolve the install dir from this script's real location (it's symlinked into ~/.local/bin)
SELF="$0"
while [[ -L "$SELF" ]]; do
  target="$(readlink "$SELF")"
  [[ "$target" == /* ]] && SELF="$target" || SELF="$(dirname "$SELF")/$target"
done
HOME_DIR="$(cd "$(dirname "$SELF")/.." && pwd)"
export HOME_DIR
REPO="UseCloakAI/CloakOS"
PORT="${CLOAK_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
PIDF="$HOME_DIR/cloak.pid"
LOGF="$HOME_DIR/logs/cloak.log"
STATUSF="$HOME_DIR/update-status.json"

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

# wire-pod goes up and down with CloakOS (macOS WirePod app). CLOAK_MANAGE_WIREPOD=0 in .env turns this off.
wp_ok() {
  [[ "$(uname -s)" == "Darwin" && -f "$HOME_DIR/bin/wirepod.sh" && "$(envval CLOAK_MANAGE_WIREPOD)" != "0" ]] || return 1
  # shellcheck source=wirepod.sh
  source "$HOME_DIR/bin/wirepod.sh"
  [[ -d "$WP_APP" ]]
}
wp_up() {
  wp_ok || return 0
  wp_running || open -g -a "$WP_APP" 2>/dev/null || return 0
  for _ in $(seq 1 60); do wp_reachable >/dev/null && return 0; sleep 0.5; done     # up to ~30 s
  echo "wire-pod is taking a while to start; CloakOS will connect once it's up."
}
wp_down() { wp_ok || return 0; wp_quit; }
up() { curl -fsS -o /dev/null "$URL/api/setup/status" 2>/dev/null; }
envval() { grep -E "^$1=" "$HOME_DIR/.env" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"' || true; }

start() {
  if running; then return 0; fi
  mkdir -p "$HOME_DIR/logs"
  cd "$HOME_DIR"
  CLOAK_HOME="$HOME_DIR" CLOAK_ENV_FILE="$HOME_DIR/.env" nohup "$HOME_DIR/venv/bin/python" -m cloakos \
    --config "$HOME_DIR/config.yaml" >>"$LOGF" 2>&1 </dev/null &
  echo $! >"$PIDF"
  for _ in $(seq 1 120); do          # up to ~30 s (first start loads the voice model)
    up && return 0
    running || { echo "CloakOS stopped during startup. See: cloakos logs"; tail -n 20 "$LOGF"; return 1; }
    sleep 0.25
  done
  echo "CloakOS is taking a while to start. See: cloakos logs"
  return 1
}

stop() {
  if running; then
    local pid; pid="$(cat "$PIDF")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done   # clean shutdown parks the robot
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDF"
}

# cloakos awake [hours|off]: keep the Mac awake (and training) with the screen off.
# No hours: until training is done (a run, or Keep going, is on). The screen goes dark right away;
# touch the trackpad to see it. Plugged in is best: on battery a closed lid still sleeps.
AWAKEF="$HOME_DIR/awake.pid"
awake_running() { [[ -f "$AWAKEF" ]] && kill -0 "$(cat "$AWAKEF")" 2>/dev/null; }
awake() {
  if [[ "$(uname -s)" != "Darwin" ]]; then echo "cloakos awake is for macOS."; return 1; fi
  if awake_running; then pkill -P "$(cat "$AWAKEF")" 2>/dev/null || true; kill "$(cat "$AWAKEF")" 2>/dev/null || true; fi
  rm -f "$AWAKEF"
  if [[ "${1:-}" == "off" ]]; then echo "Normal sleep is back."; return 0; fi
  if [[ -n "${1:-}" ]]; then
    local secs; secs="$(awk -v h="$1" 'BEGIN { if (h+0 <= 0) exit 1; printf "%d", h * 3600 }')" \
      || { echo "usage: cloakos awake [hours|off]"; return 2; }
    nohup caffeinate -ims -t "$secs" >/dev/null 2>&1 </dev/null &
    echo $! >"$AWAKEF"
    echo "Staying awake for $1 h (screen off). Undo with: cloakos awake off"
  else
    # a watcher that lives while there's training to do; caffeinate lives as long as the watcher
    nohup "$SELF" _awake-watch >/dev/null 2>&1 </dev/null &
    echo $! >"$AWAKEF"
    echo "Staying awake until training is done (screen off). Undo with: cloakos awake off"
  fi
  sleep 1
  pmset displaysleepnow 2>/dev/null || true       # dimmest there is: the screen off, the Mac still running
}

training_busy() {   # a training run going, or Keep going switched on
  curl -fsS -m 5 "$URL/api/rl/running" 2>/dev/null | grep -Eq '"running": *"' && return 0
  grep -Eq '"on": *true' "$HOME_DIR/data/keep_training.json" 2>/dev/null
}
awake_watch() {
  caffeinate -ims -w $$ &
  sleep 120                                       # give a just-started run time to show up
  while training_busy; do sleep 60; done
}

browse() {
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
  else echo "$URL"; fi
}

write_status() {     # write_status <result> <latest sha>
  printf '{"checked_at": %s, "result": "%s", "installed": "%s", "latest": "%s", "channel": "%s"}\n' \
    "$(date +%s)" "$1" "$(cat "$HOME_DIR/app/.sha" 2>/dev/null || echo unknown)" "$2" "$(channel)" >"$STATUSF"
}
pyvar() { sed -n "s/^$1 *= *\"\([^\"]*\)\".*/\1/p" | head -n1; }     # pyvar NAME < __init__.py
label() { local v="${1:-?}" n="${2:-}"; echo "v$v${n:+ (Nous v$n)}"; }
channel() { cat "$HOME_DIR/channel" 2>/dev/null || echo main; }

auto_update() {      # auto-update [--now]: install a newer Cloak (and with it any newly-pinned wire-pod)
  local force="${1:-}" latest installed was_running="" from_label to_label
  if [[ "$(envval CLOAK_AUTO_UPDATE)" == "0" && "$force" != "--now" ]]; then write_status "off" ""; return 0; fi
  if ! mkdir "$HOME_DIR/.updating" 2>/dev/null; then
    # a leftover lock from an interrupted update: clear it if it's over 10 minutes old
    if [[ -n "$(find "$HOME_DIR/.updating" -maxdepth 0 -mmin +10 2>/dev/null)" ]]; then rmdir "$HOME_DIR/.updating" 2>/dev/null || true; mkdir "$HOME_DIR/.updating" 2>/dev/null || true
    else echo "An update is already running."; return 0; fi
  fi
  trap 'rmdir "$HOME_DIR/.updating" 2>/dev/null || true' RETURN
  installed="$(cat "$HOME_DIR/app/.sha" 2>/dev/null || echo unknown)"
  latest="$(curl -fsSL -m 15 -H 'Accept: application/vnd.github.sha' \
    "https://api.github.com/repos/$REPO/commits/$(channel)" 2>/dev/null || true)"
  if [[ ! "$latest" =~ ^[0-9a-f]{40}$ ]]; then
    write_status "offline" ""; [[ "$force" == "--now" ]] && echo "Couldn't check for updates. Are you online?"; return 0
  fi
  if [[ "$latest" == "$installed" ]]; then
    write_status "current" "$latest"; [[ "$force" == "--now" ]] && echo "CloakOS is up to date: $(label "$(pyvar __version__ <"$HOME_DIR/app/cloakos/__init__.py" 2>/dev/null)" "$(pyvar NOUS_VERSION <"$HOME_DIR/app/cloakos/__init__.py" 2>/dev/null)") (${installed:0:7})."; return 0
  fi
  # never interrupt a recording; try again next time
  if running && curl -fsS -m 3 "$URL/api/health" 2>/dev/null | grep -q '"recording": *true'; then
    write_status "postponed (recording)" "$latest"
    [[ "$force" == "--now" ]] && echo "An update is ready, but CloakOS is recording. Stop the recording and run: cloakos update"
    return 0
  fi
  local init_old init_new
  init_old="$(cat "$HOME_DIR/app/cloakos/__init__.py" 2>/dev/null || true)"
  init_new="$(curl -fsSL -m 15 "https://raw.githubusercontent.com/$REPO/$latest/cloakos/__init__.py" 2>/dev/null || true)"
  from_label="$(label "$(pyvar __version__ <<<"$init_old")" "$(pyvar NOUS_VERSION <<<"$init_old")")"
  to_label="$(label "$(pyvar __version__ <<<"$init_new")" "$(pyvar NOUS_VERSION <<<"$init_new")")"
  echo "$(date '+%F %T') updating CloakOS $from_label (${installed:0:7}) -> $to_label (${latest:0:7})"
  running && was_running=1
  if curl -fsSL "https://raw.githubusercontent.com/$REPO/$latest/install.sh" \
      | CLOAK_REF="$latest" CLOAK_CHANNEL="$(channel)" CLOAK_HOME="$HOME_DIR" CLOAK_NO_OPEN=1 CLOAK_FROM_UPDATER=1 bash; then
    write_status "updated" "$latest"; [[ "$force" == "--now" ]] && echo "Updated to CloakOS $to_label (${latest:0:7})."
  else
    write_status "failed" "$latest"; [[ "$force" == "--now" ]] && echo "Update failed. See: $HOME_DIR/logs/update.log"
  fi
  if [[ -n "$was_running" ]]; then
    # an update can move an old "Cloak Vector" install to its CloakOS home; start it from there
    if [[ -x "$HOME_DIR/venv/bin/python" ]]; then start; else "$(cloakos_home)/bin/cloakctl" start; fi
  fi
}

cloakos_home() {
  if [[ "$(uname -s)" == "Darwin" ]]; then echo "$HOME/Library/Application Support/CloakOS"; else echo "$HOME/.local/share/cloakos"; fi
}

uninstall() {        # the standalone uninstaller does the work; run a copy, since it deletes this folder
  local tmp; tmp="$(mktemp)"
  if [[ -f "$HOME_DIR/bin/uninstall.sh" ]]; then cp "$HOME_DIR/bin/uninstall.sh" "$tmp"
  else curl -fsSL "https://raw.githubusercontent.com/$REPO/main/uninstall.sh" -o "$tmp"; fi
  CLOAK_HOME="$HOME_DIR" bash "$tmp" "$@"
  rm -f "$tmp"
}

main() {
  case "${1:-open}" in
    open)        wp_up; start && browse ;;
    start)       wp_up; start && echo "Cloak is running at $URL" ;;
    stop)        stop; wp_down; echo "Stopped (CloakOS and wire-pod)." ;;
    restart)     stop; wp_down; wp_up; start && echo "Restarted at $URL" ;;
    status)      if running && up; then echo "Running at $URL (pid $(cat "$PIDF"))"; else echo "Not running."; fi
                 awake_running && echo "Keeping the Mac awake (cloakos awake off to stop)." || true ;;
    logs)        tail -n 100 -f "$LOGF" ;;
    awake)       awake "${2:-}" ;;
    _awake-watch) awake_watch ;;
    update)      auto_update --now ;;
    auto-update) auto_update "${2:-}" ;;
    uninstall)   shift; uninstall "$@" ;;
    *) echo "usage: cloakos [open|start|stop|restart|status|logs|awake [hours|off]|update|uninstall]"; exit 2 ;;
  esac
}

# one line: bash has read it all before running, so an update can safely replace this file mid-run
main "$@"; exit $?
