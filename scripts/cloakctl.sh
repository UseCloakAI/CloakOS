#!/usr/bin/env bash
# cloak-vector: control the installed app.
#   open | start | stop | restart | status | logs | update | auto-update | uninstall
set -euo pipefail

# resolve the install dir from this script's real location (it's symlinked into ~/.local/bin)
SELF="$0"
while [[ -L "$SELF" ]]; do
  target="$(readlink "$SELF")"
  [[ "$target" == /* ]] && SELF="$target" || SELF="$(dirname "$SELF")/$target"
done
HOME_DIR="$(cd "$(dirname "$SELF")/.." && pwd)"
export HOME_DIR
REPO="UseCloakAI/CloakVOS"
PORT="${CLOAK_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
PIDF="$HOME_DIR/cloak.pid"
LOGF="$HOME_DIR/logs/cloak.log"
STATUSF="$HOME_DIR/update-status.json"

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
up() { curl -fsS -o /dev/null "$URL/api/setup/status" 2>/dev/null; }
envval() { grep -E "^$1=" "$HOME_DIR/.env" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"' || true; }

start() {
  if running; then return 0; fi
  mkdir -p "$HOME_DIR/logs"
  cd "$HOME_DIR"
  CLOAK_HOME="$HOME_DIR" CLOAK_ENV_FILE="$HOME_DIR/.env" nohup "$HOME_DIR/venv/bin/python" -m cloak_vector \
    --config "$HOME_DIR/config.yaml" >>"$LOGF" 2>&1 </dev/null &
  echo $! >"$PIDF"
  for _ in $(seq 1 120); do          # up to ~30 s (first start loads the voice model)
    up && return 0
    running || { echo "Cloak stopped during startup. See: cloak-vector logs"; tail -n 20 "$LOGF"; return 1; }
    sleep 0.25
  done
  echo "Cloak is taking a while to start. See: cloak-vector logs"
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

browse() {
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
  else echo "$URL"; fi
}

write_status() {     # write_status <result> <latest sha>
  printf '{"checked_at": %s, "result": "%s", "installed": "%s", "latest": "%s", "channel": "%s"}\n' \
    "$(date +%s)" "$1" "$(cat "$HOME_DIR/app/.sha" 2>/dev/null || echo unknown)" "$2" "$(channel)" >"$STATUSF"
}
channel() { cat "$HOME_DIR/channel" 2>/dev/null || echo main; }

auto_update() {      # auto-update [--now]: install a newer Cloak (and with it any newly-pinned wire-pod)
  local force="${1:-}" latest installed was_running=""
  if [[ "$(envval CLOAK_AUTO_UPDATE)" == "0" && "$force" != "--now" ]]; then write_status "off" ""; return 0; fi
  mkdir "$HOME_DIR/.updating" 2>/dev/null || { echo "update already running"; return 0; }
  trap 'rmdir "$HOME_DIR/.updating" 2>/dev/null || true' RETURN
  installed="$(cat "$HOME_DIR/app/.sha" 2>/dev/null || echo unknown)"
  latest="$(curl -fsSL -m 15 -H 'Accept: application/vnd.github.sha' \
    "https://api.github.com/repos/$REPO/commits/$(channel)" 2>/dev/null || true)"
  if [[ ! "$latest" =~ ^[0-9a-f]{40}$ ]]; then write_status "offline" ""; return 0; fi
  if [[ "$latest" == "$installed" ]]; then write_status "current" "$latest"; return 0; fi
  # never interrupt a recording; try again next time
  if running && curl -fsS -m 3 "$URL/api/health" 2>/dev/null | grep -q '"recording": *true'; then
    write_status "postponed (recording)" "$latest"; return 0
  fi
  echo "$(date '+%F %T') updating $installed -> $latest"
  running && was_running=1
  if curl -fsSL "https://raw.githubusercontent.com/$REPO/$latest/install.sh" \
      | CLOAK_REF="$latest" CLOAK_CHANNEL="$(channel)" CLOAK_HOME="$HOME_DIR" CLOAK_NO_OPEN=1 CLOAK_FROM_UPDATER=1 bash; then
    write_status "updated" "$latest"
  else
    write_status "failed" "$latest"
  fi
  [[ -n "$was_running" ]] && start || true
}

uninstall() {
  stop
  read -r -p "Remove Cloak for Vector, its settings and recordings? [y/N] " yn </dev/tty
  [[ "$yn" == [yY]* ]] || { echo "Kept."; return 0; }
  local label
  for label in org.usecloak.vector org.usecloak.vector.update; do
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/$label.plist"
  done
  if [[ -f "$HOME_DIR/wirepod.managed" ]]; then
    read -r -p "Also remove WirePod (Cloak installed it; your paired robot settings go with it)? [y/N] " yn </dev/tty
    if [[ "$yn" == [yY]* ]]; then
      # shellcheck source=wirepod.sh
      source "$HOME_DIR/bin/wirepod.sh"
      wp_quit
      rm -f "$WP_AGENT"
      rm -rf "$WP_APP" 2>/dev/null || sudo rm -rf "$WP_APP"
    fi
  fi
  rm -rf "$HOME/Applications/Cloak Vector.app" "$HOME/.local/bin/cloak-vector" "$HOME_DIR"
  echo "Removed. (uv stays in ~/.local/bin if you want it; delete it there otherwise.)"
}

main() {
  case "${1:-open}" in
    open)        start && browse ;;
    start)       start && echo "Cloak is running at $URL" ;;
    stop)        stop && echo "Stopped." ;;
    restart)     stop; start && echo "Restarted at $URL" ;;
    status)      if running && up; then echo "Running at $URL (pid $(cat "$PIDF"))"; else echo "Not running."; fi ;;
    logs)        tail -n 100 -f "$LOGF" ;;
    update)      auto_update --now ;;
    auto-update) auto_update "${2:-}" ;;
    uninstall)   uninstall ;;
    *) echo "usage: cloak-vector [open|start|stop|restart|status|logs|update|uninstall]"; exit 2 ;;
  esac
}

# one line: bash has read it all before running, so an update can safely replace this file mid-run
main "$@"; exit $?
