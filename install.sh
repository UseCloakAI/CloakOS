#!/usr/bin/env bash
# Cloak for Vector: one-command installer for macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakOS/main/install.sh | bash
#
# Installs everything into ~/Library/Application Support/Cloak Vector (Python via uv,
# the app, the voice model), adds "Cloak Vector" to ~/Applications and a `cloak-vector`
# command, then opens the setup wizard in your browser. Re-run it any time to update;
# your settings and recordings are kept.
#
# Env knobs: CLOAK_REF (branch/tag, default main) · CLOAK_SOURCE (local checkout, for dev)
#            CLOAK_HOME (install dir) · CLOAK_NO_OPEN=1 (don't start/open at the end)
set -euo pipefail

REPO="UseCloakAI/CloakOS"
REF="${CLOAK_REF:-main}"
PORT=8765
OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
  HOME_DIR="${CLOAK_HOME:-$HOME/Library/Application Support/Cloak Vector}"
else
  HOME_DIR="${CLOAK_HOME:-$HOME/.local/share/cloak-vector}"
fi
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
  printf '%s   ██████ ██       ██████   █████  ██   ██%s\n' "$ACC" "$RST"
  printf '%s  ██      ██      ██    ██ ██   ██ ██  ██ %s\n' "$ACC" "$RST"
  printf '%s  ██      ██      ██    ██ ███████ █████  %s\n' "$ACC" "$RST"
  printf '%s  ██      ██      ██    ██ ██   ██ ██  ██ %s\n' "$ACC" "$RST"
  printf '%s   ██████ ███████  ██████  ██   ██ ██   ██%s\n' "$ACC" "$RST"
  printf '\n  %sF O R   V E C T O R%s   %sa living creature brain · usecloak.org%s\n' "$YEL$INK" "$RST" "$DIM" "$RST"
  printf '  %s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n\n' "$DIM" "$RST"
}

STEP=0
TOTAL=7
# step "Label" cmd...   runs cmd quietly with a spinner; on failure shows the log tail
step() {
  local label="$1"; shift
  STEP=$((STEP + 1))
  local n; n=$(printf '%02d' "$STEP")
  printf '%s' "$HIDE"
  ( "$@" ) >>"$LOG" 2>&1 &
  local pid=$! i=0 t0=$SECONDS
  local frames=('◰' '◳' '◲' '◱')
  while kill -0 "$pid" 2>/dev/null; do
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

note() { printf '      %s%s%s\n' "$DIM" "$1" "$RST"; }

# ── steps ────────────────────────────────────────────────────────────
need_uv() {
  if command -v uv >/dev/null 2>&1; then return; fi
  if [[ -x "$HOME/.local/bin/uv" ]]; then return; fi
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
}
uv_bin() { command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv"; }

get_app() {
  local tmp; tmp="$(mktemp -d)"
  if [[ -n "${CLOAK_SOURCE:-}" ]]; then
    tar -C "$CLOAK_SOURCE" --exclude .git --exclude data --exclude models --exclude '.venv' --exclude '__pycache__' -cf - . | tar -C "$tmp" -xf -
  else
    curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$REF" | tar -xz -C "$tmp" --strip-components 1
  fi
  rm -rf "$HOME_DIR/app"
  mv "$tmp" "$HOME_DIR/app"
}

python_env() {
  local uv; uv="$(uv_bin)"
  "$uv" venv --quiet --allow-existing --python 3.12 "$HOME_DIR/venv"
  "$uv" pip install --quiet --python "$HOME_DIR/venv/bin/python" "$HOME_DIR/app[kokoro]"
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
  [[ -f "$HOME_DIR/.env" ]] || { cp "$HOME_DIR/app/.env.example" "$HOME_DIR/.env"; chmod 600 "$HOME_DIR/.env"; }
  mkdir -p "$HOME_DIR/data" "$HOME_DIR/bin"
  install -m 755 "$HOME_DIR/app/scripts/cloakctl.sh" "$HOME_DIR/bin/cloakctl"
}

launchers() {
  mkdir -p "$BIN_DIR"
  ln -sf "$HOME_DIR/bin/cloakctl" "$BIN_DIR/cloak-vector"
  [[ "$OS" == "Darwin" ]] || return 0
  local app="$HOME/Applications/Cloak Vector.app"
  rm -rf "$app"
  mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cat >"$app/Contents/MacOS/Cloak Vector" <<EOF
#!/bin/bash
exec "$HOME_DIR/bin/cloakctl" open
EOF
  chmod 755 "$app/Contents/MacOS/Cloak Vector"
  cat >"$app/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Cloak Vector</string>
  <key>CFBundleDisplayName</key><string>Cloak Vector</string>
  <key>CFBundleIdentifier</key><string>org.usecloak.vector</string>
  <key>CFBundleExecutable</key><string>Cloak Vector</string>
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

# ── go ───────────────────────────────────────────────────────────────
banner
if [[ "$OS" != "Darwin" && -z "${CLOAK_ALLOW_LINUX:-}" ]]; then
  printf '  %sThis installer is for macOS.%s Set CLOAK_ALLOW_LINUX=1 to try it anyway.\n\n' "$RED" "$RST"
  exit 1
fi
mkdir -p "$HOME_DIR/logs"
: >"$LOG"
VERB="Installing"; [[ -d "$HOME_DIR/venv" ]] && VERB="Updating"
printf '  %s%s into%s %s\n\n' "$DIM" "$VERB" "$RST" "${HOME_DIR/#$HOME/~}"

"$HOME_DIR/bin/cloakctl" stop >/dev/null 2>&1 || true

step "Python toolchain (uv)"       need_uv
step "Download Cloak for Vector"   get_app
step "Install the brain"           python_env
step "Voice model (~120 MB)"       voice_model
step "Settings"                    settings
step "App + cloak-vector command"  launchers
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
note "Open it any time:   Cloak Vector in Applications, or  cloak-vector open"
note "Other commands:     cloak-vector stop | status | logs | update | uninstall"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) note "Add ~/.local/bin to your PATH to use cloak-vector from any terminal." ;; esac
printf '\n'
