#!/usr/bin/env bash
# CloakOS (Vector Edition): uninstaller for macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakVOS/main/uninstall.sh | bash
#   cloakos uninstall                       (same thing, from an installed copy)
#
# Removes everything CloakOS installed: the app, the cloakos command, the background jobs
# (start at login, auto-update), the install folder (Python env, voice model, settings),
# WirePod (app, login item, its settings) and uv + the Python it downloaded, each only if
# CloakOS installed it. Older "Cloak Vector" installs are cleaned up too.
# Your recordings are moved to ~/Documents/CloakOS Recordings, not deleted.
#   --yes                  don't ask for confirmation
#   --delete-recordings    delete recordings instead of keeping them
set -euo pipefail

YES="" DEL_REC=""
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=1 ;;
    --delete-recordings) DEL_REC=1 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
  HOMES=("$HOME/Library/Application Support/CloakOS" "$HOME/Library/Application Support/Cloak Vector")
else
  HOMES=("$HOME/.local/share/cloakos" "$HOME/.local/share/cloak-vector")
fi
[[ -n "${CLOAK_HOME:-}" ]] && HOMES=("$CLOAK_HOME" "${HOMES[@]}")
BIN_DIR="$HOME/.local/bin"
KEEP_DIR="$HOME/Documents/CloakOS Recordings"
LABELS=(org.usecloak.cloakos org.usecloak.cloakos.update org.usecloak.vector org.usecloak.vector.update)
APPS=("$HOME/Applications/CloakOS.app" "$HOME/Applications/Cloak Vector.app")
CMDS=("$BIN_DIR/cloakos" "$BIN_DIR/cloak-vector")
WP_APP="/Applications/WirePod.app"
WP_AGENT="$HOME/Library/LaunchAgents/WirePod.agent.plist"
WP_DATA="$HOME/Library/Application Support/wire-pod"

if [[ -t 1 ]]; then
  ACC=$'\033[38;5;166m'; YEL=$'\033[38;5;221m'; INK=$'\033[1m'; DIM=$'\033[2m'; OK=$'\033[38;5;71m'; RST=$'\033[0m'
else
  ACC=""; YEL=""; INK=""; DIM=""; OK=""; RST=""
fi

say()  { printf '  %s■%s %s\n' "$ACC" "$RST" "$1"; }
done_() { printf '    %s%s%s\n' "$OK" "$1" "$RST"; }
note() { printf '    %s%s%s\n' "$DIM" "$1" "$RST"; }

# ask "question" default(y|n) -> 0 for yes. Reads the keyboard even when piped from curl.
ask() {
  local q="$1" def="$2" yn
  [[ -n "$YES" ]] && return 0                                   # --yes: everything is a yes
  if ! { : </dev/tty; } 2>/dev/null; then [[ "$def" == y ]]; return; fi   # no keyboard: take the safe default
  local hint="[y/N]"; [[ "$def" == y ]] && hint="[Y/n]"
  read -r -p "  $q $hint " yn </dev/tty || yn=""
  [[ -z "$yn" ]] && yn="$def"
  [[ "$yn" == [yY]* ]]
}

# only ever delete a folder that is recognisably a CloakOS install
is_install() { [[ -n "$1" && -d "$1" && ( -x "$1/bin/cloakctl" || -d "$1/venv" || -f "$1/app/compat.json" ) ]]; }

printf '\n  %sCloakOS%s %sVector Edition · uninstall%s\n' "$ACC$INK" "$RST" "$YEL" "$RST"
printf '  %s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n\n' "$DIM" "$RST"

found=()
for h in "${HOMES[@]}"; do is_install "$h" && [[ ! " ${found[*]-} " == *" $h "* ]] && found+=("$h"); done
leftovers=0
for p in "${APPS[@]}" "${CMDS[@]}"; do [[ -e "$p" || -L "$p" ]] && leftovers=1; done
for l in "${LABELS[@]}"; do [[ -f "$HOME/Library/LaunchAgents/$l.plist" ]] && leftovers=1; done
if [[ ${#found[@]} -eq 0 && $leftovers -eq 0 ]]; then
  say "CloakOS isn't installed. Nothing to do."
  printf '\n'; exit 0
fi

for h in "${found[@]+"${found[@]}"}"; do note "Found: ${h/#"$HOME"/\~}"; done
wp_managed="" uv_managed=""
for h in "${found[@]+"${found[@]}"}"; do
  [[ -f "$h/wirepod.managed" ]] && wp_managed=1
  [[ -f "$h/uv.managed" ]] && uv_managed=1
done
what="CloakOS${wp_managed:+, WirePod}${uv_managed:+, uv}"
note "This removes $what and everything CloakOS set up."
[[ -z "$DEL_REC" ]] && note "Recordings are kept in ~/Documents/CloakOS Recordings."
ask "Remove CloakOS from this Mac?" n || { printf '\n  Kept everything.\n\n'; exit 0; }
printf '\n'

# 1 · stop it
say "Stopping CloakOS"
for h in "${found[@]+"${found[@]}"}"; do
  if [[ -f "$h/cloak.pid" ]]; then
    pid="$(cat "$h/cloak.pid")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done   # lets it park the robot
    kill -9 "$pid" 2>/dev/null || true
  fi
done
done_ "stopped"

# 2 · background jobs
say "Removing start-at-login and auto-update"
for l in "${LABELS[@]}"; do
  [[ "$OS" == "Darwin" ]] && launchctl bootout "gui/$(id -u)/$l" 2>/dev/null || true
  rm -f "$HOME/Library/LaunchAgents/$l.plist"
done
done_ "removed"

# 3 · app + command
say "Removing the app and the cloakos command"
for p in "${APPS[@]}" "${CMDS[@]}"; do rm -rf "$p"; done
done_ "removed"

# 4 · recordings: your training data, not something CloakOS installed, so they're kept by default
kept_to=""
for h in "${found[@]+"${found[@]}"}"; do
  ep="$h/data/episodes"
  [[ -d "$ep" && -n "$(ls -A "$ep" 2>/dev/null)" ]] || continue
  [[ -n "$DEL_REC" ]] && continue
  mkdir -p "$KEEP_DIR"
  for e in "$ep"/*; do
    dest="$KEEP_DIR/$(basename "$e")"
    [[ -e "$dest" ]] && dest="$dest-$(date +%s)"
    mv "$e" "$dest"
  done
  kept_to="$KEEP_DIR"
done
[[ -n "$kept_to" ]] && { say "Recordings kept"; done_ "${kept_to/#"$HOME"/\~}"; }

# 5 · WirePod, if CloakOS installed it (a wire-pod you set up yourself is left alone)
if [[ -n "$wp_managed" && "$OS" == "Darwin" ]]; then
  say "Removing WirePod"
  osascript -e 'tell application "WirePod" to quit' >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do pgrep -x WirePod >/dev/null 2>&1 || break; sleep 0.25; done
  pkill -x WirePod 2>/dev/null || true
  launchctl bootout "gui/$(id -u)/WirePod.agent" 2>/dev/null || true
  rm -f "$WP_AGENT"
  if [[ -d "$WP_APP" ]]; then rm -rf "$WP_APP" 2>/dev/null || sudo rm -rf "$WP_APP"; fi
  rm -rf "$WP_DATA"
  done_ "removed (app, login item, settings)"
elif [[ -d "$WP_APP" ]]; then
  note "WirePod was already on this Mac before CloakOS, so it stays."
fi

# 6 · the install folder(s)
say "Removing the install folder"
for h in "${found[@]+"${found[@]}"}"; do
  is_install "$h" && rm -rf "$h" && done_ "${h/#"$HOME"/\~}"
done

# 7 · uv + the Python it downloaded, if CloakOS installed uv
if [[ -n "$uv_managed" ]]; then
  say "Removing uv and its Python"
  rm -f "$BIN_DIR/uv" "$BIN_DIR/uvx" "$BIN_DIR/uvw"
  rm -rf "$HOME/.local/share/uv" "$HOME/.cache/uv" "$HOME/Library/Caches/uv"
  done_ "removed"
fi

printf '\n  %s■%s %sCloakOS is uninstalled.%s\n' "$ACC" "$RST" "$INK" "$RST"
note "To install again:  curl -fsSL https://raw.githubusercontent.com/UseCloakAI/CloakVOS/main/install.sh | bash"
printf '\n'
