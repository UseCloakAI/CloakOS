#!/usr/bin/env bash
# cloak-vector: control the installed app.  open | start | stop | restart | status | logs | update | uninstall
set -euo pipefail

# resolve the install dir from this script's real location (it's symlinked into ~/.local/bin)
SELF="$0"
while [[ -L "$SELF" ]]; do
  target="$(readlink "$SELF")"
  [[ "$target" == /* ]] && SELF="$target" || SELF="$(dirname "$SELF")/$target"
done
HOME_DIR="$(cd "$(dirname "$SELF")/.." && pwd)"
PORT="${CLOAK_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
PIDF="$HOME_DIR/cloak.pid"
LOGF="$HOME_DIR/logs/cloak.log"

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
up() { curl -fsS -o /dev/null "$URL/api/setup/status" 2>/dev/null; }

start() {
  if running; then return 0; fi
  mkdir -p "$HOME_DIR/logs"
  cd "$HOME_DIR"
  CLOAK_ENV_FILE="$HOME_DIR/.env" nohup "$HOME_DIR/venv/bin/python" -m cloak_vector --config "$HOME_DIR/config.yaml" \
    >>"$LOGF" 2>&1 </dev/null &
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

case "${1:-open}" in
  open)      start && browse ;;
  start)     start && echo "Cloak is running at $URL" ;;
  stop)      stop && echo "Stopped." ;;
  restart)   stop; start && echo "Restarted at $URL" ;;
  status)    if running && up; then echo "Running at $URL (pid $(cat "$PIDF"))"; else echo "Not running."; fi ;;
  logs)      tail -n 100 -f "$LOGF" ;;
  update)    curl -fsSL "https://raw.githubusercontent.com/UseCloakAI/CloakOS/${CLOAK_REF:-main}/install.sh" | bash ;;
  uninstall)
    stop
    read -r -p "Remove Cloak for Vector, its settings and recordings? [y/N] " yn </dev/tty
    [[ "$yn" == [yY]* ]] || { echo "Kept."; exit 0; }
    rm -rf "$HOME/Applications/Cloak Vector.app" "$HOME/.local/bin/cloak-vector" "$HOME_DIR"
    echo "Removed. (uv stays in ~/.local/bin if you want it; delete it there otherwise.)" ;;
  *) echo "usage: cloak-vector [open|start|stop|restart|status|logs|update|uninstall]"; exit 2 ;;
esac
