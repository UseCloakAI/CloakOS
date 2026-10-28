#!/usr/bin/env bash
# CloakOS (Vector Edition): one-command installer for macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakOS/main/install.sh | bash
#
# Installs everything into ~/Library/Application Support/CloakOS (Python via uv, the app,
# the voice model), adds CloakOS to ~/Applications and a `cloakos` command, then opens
# CloakOS in your browser. Re-run it any time to update; your settings and recordings are
# kept. Older "Cloak Vector" installs are moved over automatically.
#
# Remove it with:  curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakOS/main/uninstall.sh | bash
#
# It also installs wire-pod (the official WirePod app) if none is on your network, and
# sets up background updates: CloakOS follows its channel (main); wire-pod only ever moves
# to the version pinned in compat.json, i.e. one we've deemed compatible.
#
# Env knobs: CLOAK_REF (branch/tag/sha, default main) · CLOAK_CHANNEL (what auto-update follows)
#            CLOAK_SOURCE (local checkout, for dev) · CLOAK_HOME (install dir)
#            CLOAK_NO_OPEN=1 (don't start/open at the end) · CLOAK_WIREPOD=skip · CLOAK_EXTRAS (default kokoro,rl,ml)
set -euo pipefail

REPO="UseCloakAI/CloakOS"
REF="${CLOAK_REF:-main}"
PORT=8765
OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
  NEW_HOME="$HOME/Library/Application Support/CloakOS"
  LEGACY_HOME="$HOME/Library/Application Support/Cloak Vector"
else
  NEW_HOME="$HOME/.local/share/cloakos"
  LEGACY_HOME="$HOME/.local/share/cloak-vector"
fi
HOME_DIR="${CLOAK_HOME:-$NEW_HOME}"
# an updater running from an old "Cloak Vector" install still passes its old home: move to the new one
[[ "$HOME_DIR" == "$LEGACY_HOME" ]] && HOME_DIR="$NEW_HOME"
BIN_DIR="$HOME/.local/bin"
LOG="$HOME_DIR/logs/install.log"
MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

# ── look ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  ACC=$'\033[38;5;166m'; YEL=$'\033[38;5;221m'; INK=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[38;5;160m'
  OK=$'\033[38;5;71m'; RST=$'\033[0m'; HIDE=$'\033[?25l'; SHOW=$'\033[?25h'
else
  ACC=""; YEL=""; INK=""; DIM=""; RED=""; OK=""; RST=""; HIDE=""; SHOW=""
fi
trap 'printf "%s" "$SHOW"' EXIT

banner() {
  printf '\n'
  printf '%s   ██████ ██       ██████   █████  ██   ██ %s ██████  ███████%s\n' "$ACC" "$YEL" "$RST"
  printf '%s  ██      ██      ██    ██ ██   ██ ██  ██  %s██    ██ ██     %s\n' "$ACC" "$YEL" "$RST"
  printf '%s  ██      ██      ██    ██ ███████ █████   %s██    ██ ███████%s\n' "$ACC" "$YEL" "$RST"
  printf '%s  ██      ██      ██    ██ ██   ██ ██  ██  %s██    ██      ██%s\n' "$ACC" "$YEL" "$RST"
  printf '%s   ██████ ███████  ██████  ██   ██ ██   ██ %s ██████  ███████%s\n' "$ACC" "$YEL" "$RST"
  printf '\n  %sV E C T O R   E D I T I O N%s   %sa living creature brain · usecloak.org%s\n' "$YEL$INK" "$RST" "$DIM" "$RST"
  printf '  %s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n\n' "$DIM" "$RST"
}

STEP=0
TOTAL=9
# step "Label" cmd...   runs cmd quietly with a spinner; on failure shows the log tail
step() {
  local label="$1"; shift
  STEP=$((STEP + 1))
  local n; n=$(printf '%02d' "$STEP")
  printf '%s' "$HIDE"
  ( "$@" ) >>"$LOG" 2>&1 &
  local pid=$! i=0 t0=$SECONDS
  local frames=('◰' '◳' '◲' '◱')
  while [[ -t 1 ]] && kill -0 "$pid" 2>/dev/null; do     # spinner only on a real terminal
    printf '\r  %s%s%s  %s%-34s%s %s%s %ss%s' "$DIM" "$n/0$TOTAL" "$RST" "$INK" "$label" "$RST" "$ACC" "${frames[i++ % 4]}" "$((SECONDS - t0))" "$RST"
    sleep 0.12
  done
  if wait "$pid"; then
    printf '\r  %s%s%s  %-34s %s■ done%s %s%ss%s   \n' "$DIM" "$n/0$TOTAL" "$RST" "$label" "$OK" "$RST" "$DIM" "$((SECONDS - t0))" "$RST"
  else
    printf '\r  %s%s%s  %-34s %s■ failed%s        \n\n' "$DIM" "$n/0$TOTAL" "$RST" "$label" "$RED" "$RST"
    printf '%s' "$SHOW"
    printf '  %sLast lines of %s:%s\n' "$DIM" "$LOG" "$RST"
    tail -n 15 "$LOG" | sed 's/^/    /'
    printf '\n  Something went wrong. Fix it and run the installer again; it picks up where it left off.\n\n'
    exit 1
  fi
  printf '%s' "$SHOW"
}

tilde() { case "$1" in "$HOME"*) printf '~%s' "${1#"$HOME"}" ;; *) printf '%s' "$1" ;; esac; }   # bash 3.2-safe ~/path

note() { printf '      %s%s%s\n' "$DIM" "$1" "$RST"; }

# ── steps ────────────────────────────────────────────────────────────
need_uv() {
  if command -v uv >/dev/null 2>&1; then return; fi
  if [[ -x "$HOME/.local/bin/uv" ]]; then return; fi
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  touch "$HOME_DIR/uv.managed"          # so the uninstaller knows uv is ours to remove
}
uv_bin() { command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv"; }

resolve_sha() {       # the exact commit we install, so the updater can tell when there's something new
  if [[ "$REF" =~ ^[0-9a-f]{40}$ ]]; then echo "$REF"; return; fi
  curl -fsSL -m 10 -H 'Accept: application/vnd.github.sha' "https://api.github.com/repos/$REPO/commits/$REF" 2>/dev/null || true
}

get_app() {
  local tmp sha; tmp="$(mktemp -d)"
  if [[ -n "${CLOAK_SOURCE:-}" ]]; then
    tar -C "$CLOAK_SOURCE" --exclude .git --exclude data --exclude models --exclude '.venv' --exclude '__pycache__' -cf - . | tar -C "$tmp" -xf -
    sha="local"
  else
    sha="$(resolve_sha)"
    curl -fsSL "https://codeload.github.com/$REPO/tar.gz/${sha:-$REF}" | tar -xz -C "$tmp" --strip-components 1
  fi
  echo "${sha:-unknown}" >"$tmp/.sha"
  rm -rf "$HOME_DIR/app"
  mv "$tmp" "$HOME_DIR/app"
  # the channel auto-update follows: explicit, else the branch you installed from, else keep
  if [[ -n "${CLOAK_CHANNEL:-}" ]]; then echo "$CLOAK_CHANNEL" >"$HOME_DIR/channel"
  elif [[ ! "$REF" =~ ^[0-9a-f]{40}$ ]]; then echo "$REF" >"$HOME_DIR/channel"; fi
}

python_env() {
  local uv; uv="$(uv_bin)"
  "$uv" venv --quiet --allow-existing --python 3.12 "$HOME_DIR/venv"
  # AI extras: rl = training in the simulator, ml = camera depth + object detection for the 3D map
  local extras="${CLOAK_EXTRAS:-kokoro,rl,ml}"
  if ! "$uv" pip install --quiet --python "$HOME_DIR/venv/bin/python" "$HOME_DIR/app[$extras]"; then
    echo "    (couldn't install the AI extras; continuing with the basics)"
    "$uv" pip install --quiet --python "$HOME_DIR/venv/bin/python" "$HOME_DIR/app[kokoro]"
  fi
  "$uv" pip uninstall --quiet --python "$HOME_DIR/venv/bin/python" cloak-vector 2>/dev/null || true   # pre-CloakOS name
}

voice_model() {
  mkdir -p "$HOME_DIR/models"
  local f
  for f in kokoro-v1.0.int8.onnx voices-v1.0.bin; do
    if [[ ! -s "$HOME_DIR/models/$f" ]]; then
      curl -fsSL -o "$HOME_DIR/models/$f.part" "$MODEL_URL/$f"
      mv "$HOME_DIR/models/$f.part" "$HOME_DIR/models/$f"
    fi
  done
}

settings() {
  # first install: start from the shipped config; updates keep yours
  [[ -f "$HOME_DIR/config.yaml" ]] || cp "$HOME_DIR/app/config.yaml" "$HOME_DIR/config.yaml"
  # settings we changed the default of: update them only if they're still exactly what we shipped
  sed -i.bak 's/^  control_priority: default   # default | override (also ignores cliffs\/low battery: careful)$/  control_priority: override  # override = stock Vector behaviors can'"'"'t take over (CloakOS does the edge stop) | default/' "$HOME_DIR/config.yaml" && rm -f "$HOME_DIR/config.yaml.bak"
  sed -i.bak 's/^  stock_blend: 0\.25 /  stock_blend: 0.0  /' "$HOME_DIR/config.yaml" && rm -f "$HOME_DIR/config.yaml.bak"
  sed -i.bak -e 's/^  chirp_volume: 70$/  chirp_volume: 100/' -e 's/^  sfx_volume: 60$/  sfx_volume: 100/' -e '/^voice:/,/^[a-z]/s/^  volume: 75$/  volume: 100/' \
    -e 's/^  eye_color: "#D44D2A"$/  eye_color: "#FFFFFF"/' -e 's/^  eye_fps: 8$/  eye_fps: 6/' -e 's/^  eye_fps: 10 .*$/  eye_fps: 6            # frames sent to Vector per second (max 8: more crashes its face\/sound process, error 800)/' \
    -e '/^    cliff: *{hint: "you almost fell off an edge"/d' \
    -e 's/^  energy_active: -0\.0015$/  energy_active: -0.004/' -e 's/^  energy_rest: 0\.006$/  energy_rest: 0.007/' "$HOME_DIR/config.yaml" && rm -f "$HOME_DIR/config.yaml.bak"
  [[ -f "$HOME_DIR/.env" ]] || { cp "$HOME_DIR/app/.env.example" "$HOME_DIR/.env"; chmod 600 "$HOME_DIR/.env"; }
  mkdir -p "$HOME_DIR/data" "$HOME_DIR/bin"
  install -m 755 "$HOME_DIR/app/scripts/cloakctl.sh" "$HOME_DIR/bin/cloakctl"
  install -m 644 "$HOME_DIR/app/scripts/wirepod.sh" "$HOME_DIR/bin/wirepod.sh"
  install -m 755 "$HOME_DIR/app/uninstall.sh" "$HOME_DIR/bin/uninstall.sh"
}

wirepod() {
  # shellcheck source=scripts/wirepod.sh
  source "$HOME_DIR/bin/wirepod.sh"
  wp_ensure
}

agent() {            # agent <label> <plist body>: write a LaunchAgent, (re)load it only if it changed
  local label="$1" body="$2" plist="$HOME/Library/LaunchAgents/$1.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  if [[ -f "$plist" ]] && [[ "$(cat "$plist")" == "$body" ]]; then return 0; fi
  printf '%s' "$body" >"$plist"
  # never reload the updater from inside an update: that would kill the update that's running
  [[ -n "${CLOAK_FROM_UPDATER:-}" ]] && return 0
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  [[ "${3:-}" == "load" ]] && launchctl bootstrap "gui/$(id -u)" "$plist" 2>/dev/null || true
}

background() {
  [[ "$OS" == "Darwin" ]] || return 0
  local head='<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>'
  # Cloak wakes up when you log in
  agent org.usecloak.cloakos "$head
  <key>Label</key><string>org.usecloak.cloakos</string>
  <key>ProgramArguments</key><array><string>$HOME_DIR/bin/cloakctl</string><string>start</string></array>
  <key>RunAtLoad</key><true/>
  <key>AbandonProcessGroup</key><true/>
</dict></plist>"
  # checks for updates every 6 hours (Cloak, and wire-pod once a new version is deemed compatible)
  agent org.usecloak.cloakos.update "$head
  <key>Label</key><string>org.usecloak.cloakos.update</string>
  <key>ProgramArguments</key><array><string>$HOME_DIR/bin/cloakctl</string><string>auto-update</string></array>
  <key>StartInterval</key><integer>21600</integer>
  <key>AbandonProcessGroup</key><true/>
  <key>StandardOutPath</key><string>$HOME_DIR/logs/update.log</string>
  <key>StandardErrorPath</key><string>$HOME_DIR/logs/update.log</string>
</dict></plist>" load
}

launchers() {
  mkdir -p "$BIN_DIR"
  ln -sf "$HOME_DIR/bin/cloakctl" "$BIN_DIR/cloakos"
  [[ "$OS" == "Darwin" ]] || return 0
  local app="$HOME/Applications/CloakOS.app"
  rm -rf "$app"
  mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cat >"$app/Contents/MacOS/CloakOS" <<EOF
#!/bin/bash
exec "$HOME_DIR/bin/cloakctl" open
EOF
  chmod 755 "$app/Contents/MacOS/CloakOS"
  cat >"$app/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>CloakOS</string>
  <key>CFBundleDisplayName</key><string>CloakOS</string>
  <key>CFBundleIdentifier</key><string>org.usecloak.cloakos</string>
  <key>CFBundleExecutable</key><string>CloakOS</string>
  <key>CFBundleGetInfoString</key><string>CloakOS Vector Edition</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict></plist>
EOF
  local iconset; iconset="$(mktemp -d)/AppIcon.iconset"
  "$HOME_DIR/venv/bin/python" "$HOME_DIR/app/scripts/make_icon.py" "$iconset"
  iconutil -c icns "$iconset" -o "$app/Contents/Resources/AppIcon.icns"
  touch "$app"
}

migrate_legacy() {   # an older "Cloak Vector" install: move it (settings, recordings, models) to CloakOS
  [[ -d "$LEGACY_HOME" ]] || return 0
  [[ -x "$LEGACY_HOME/bin/cloakctl" ]] && "$LEGACY_HOME/bin/cloakctl" stop >/dev/null 2>&1 || true
  local label
  for label in org.usecloak.vector org.usecloak.vector.update; do
    # don't boot out the job we may be running inside (an update); it just won't come back
    [[ -z "${CLOAK_FROM_UPDATER:-}" ]] && launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/$label.plist"
  done
  rm -rf "$HOME/Applications/Cloak Vector.app" "$BIN_DIR/cloak-vector"
  if [[ -d "$NEW_HOME" ]]; then
    printf '  %sBoth "Cloak Vector" and CloakOS folders exist; keeping CloakOS, leaving the old one at %s%s\n' "$DIM" "$LEGACY_HOME" "$RST"
  else
    mv "$LEGACY_HOME" "$NEW_HOME"
    printf '  %sMoved your Cloak Vector install to CloakOS.%s\n' "$DIM" "$RST"
  fi
}

# ── go ───────────────────────────────────────────────────────────────
banner
if [[ "$OS" != "Darwin" && -z "${CLOAK_ALLOW_LINUX:-}" ]]; then
  printf '  %sThis installer is for macOS.%s Set CLOAK_ALLOW_LINUX=1 to try it anyway.\n\n' "$RED" "$RST"
  exit 1
fi
[[ -z "${CLOAK_HOME:-}" || "${CLOAK_HOME:-}" == "$LEGACY_HOME" ]] && migrate_legacy
mkdir -p "$HOME_DIR/logs"
: >"$LOG"
VERB="Installing"; [[ -d "$HOME_DIR/venv" ]] && VERB="Updating"
printf '  %s%s into%s %s\n\n' "$DIM" "$VERB" "$RST" "$(tilde "$HOME_DIR")"

"$HOME_DIR/bin/cloakctl" stop >/dev/null 2>&1 || true

step "Python toolchain (uv)"       need_uv
step "Download CloakOS"            get_app
step "Install the brain"           python_env
step "Voice model (~120 MB)"       voice_model
step "Settings"                    settings
if [[ "${CLOAK_WIREPOD:-}" == "skip" ]]; then STEP=$((STEP + 1)); else
step "wire-pod"                    wirepod
fi
step "App + cloakos command"       launchers
step "Auto-start + auto-update"    background
if [[ -z "${CLOAK_NO_OPEN:-}" ]]; then
  step "Wake it up"                "$HOME_DIR/bin/cloakctl" start
else
  STEP=$((STEP + 1))
fi

printf '\n  %s■%s %sReady.%s ' "$ACC" "$RST" "$INK" "$RST"
if [[ -z "${CLOAK_NO_OPEN:-}" ]]; then
  printf 'Finish setup in your browser: %shttp://127.0.0.1:%s%s\n\n' "$ACC" "$PORT" "$RST"
  "$HOME_DIR/bin/cloakctl" open >/dev/null 2>&1 || true
else
  printf '\n\n'
fi
note "Open it any time:   CloakOS in Applications, or  cloakos open"
note "Updates install themselves in the background. Turn off in Settings."
note "Other commands:     cloakos stop | status | logs | update | uninstall"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) note "Add ~/.local/bin to your PATH to use cloakos from any terminal." ;; esac
printf '\n'
