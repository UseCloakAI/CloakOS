# wire-pod helpers for the installer and updater (sourced, macOS).
#
# Cloak only manages a wire-pod it installed itself (marker: $HOME_DIR/wirepod.managed).
# It installs the official WirePod app (github.com/kercre123/WirePod) and only ever moves
# it to the version pinned in compat.json, never straight to "latest".

WP_APP="/Applications/WirePod.app"
WP_CONF_DIR="$HOME/Library/Application Support/wire-pod"
WP_AGENT="$HOME/Library/LaunchAgents/WirePod.agent.plist"

wp_pinned() {        # pinned macOS version from compat.json
  "$HOME_DIR/venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wirepod"]["macos"]["pinned"])' "$1"
}

wp_installed_version() {
  [[ -f "$WP_APP/Contents/Resources/version" ]] && tr -d '[:space:]' <"$WP_APP/Contents/Resources/version" || true
}

wp_managed() { [[ -f "$HOME_DIR/wirepod.managed" ]]; }

wp_reachable() {     # any wire-pod answering on this Mac or the network
  local u
  for u in http://localhost:8080 http://escapepod.local:8080; do
    curl -fsS -m 3 -o /dev/null "$u/api-sdk/get_sdk_info" 2>/dev/null && { echo "$u"; return 0; }
    # an un-set-up wire-pod has no bots yet and returns 500 here, but it's still there
    [[ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$u/api-sdk/get_sdk_info" 2>/dev/null)" == "500" ]] && { echo "$u"; return 0; }
  done
  return 1
}

wp_running() { pgrep -x WirePod >/dev/null 2>&1; }

wp_quit() {
  wp_running || return 0
  osascript -e 'tell application "WirePod" to quit' >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do wp_running || return 0; sleep 0.25; done
  pkill -x WirePod 2>/dev/null || true
}

wp_seed_config() {   # answer WirePod's first-run question (run at login: yes) so it starts silently
  local ver="$1"
  mkdir -p "$WP_CONF_DIR" "$HOME/Library/LaunchAgents"
  if [[ ! -f "$WP_CONF_DIR/wire-pod-conf.json" ]]; then
    printf '{"wsport":"8080","runatstartup":true,"runtimepath":"%s","version":"%s","needsrestart":false,"lastrunningpid":0,"firststartup":false,"nopodwarn":false}\n' \
      "$WP_APP/Contents/MacOS/../Frameworks" "$ver" >"$WP_CONF_DIR/wire-pod-conf.json"
  fi
  # same login item WirePod itself writes when you answer "Yes"
  cat >"$WP_AGENT" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>WirePod.agent</string>
	<key>ProgramArguments</key>
	<array>
		<string>$WP_APP/Contents/MacOS/WirePod</string>
		<string>-d</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
</dict>
</plist>
EOF
}

wp_install() {       # wp_install v1.2.18 : download the DMG, put WirePod.app in /Applications, start it
  local ver="$1" tmp mnt
  tmp="$(mktemp -d)"; mnt="$tmp/mnt"
  curl -fsSL -o "$tmp/WirePod.dmg" "https://github.com/kercre123/WirePod/releases/download/$ver/WirePod-$ver.dmg"
  mkdir -p "$mnt"
  hdiutil attach -nobrowse -quiet -readonly -mountpoint "$mnt" "$tmp/WirePod.dmg"
  wp_quit
  if [[ -w /Applications ]]; then
    rm -rf "$WP_APP" && ditto "$mnt/WirePod.app" "$WP_APP"
  else
    sudo rm -rf "$WP_APP" && sudo ditto "$mnt/WirePod.app" "$WP_APP"
  fi
  hdiutil detach -quiet "$mnt" || true
  rm -rf "$tmp"
  xattr -dr com.apple.quarantine "$WP_APP" 2>/dev/null || true
  wp_seed_config "$ver"
  echo "$ver" >"$HOME_DIR/wirepod.managed"
  open -g -a "$WP_APP"
}

wp_lang() {          # macOS locale -> a wire-pod speech language (falls back to en-US)
  local loc; loc="$(defaults read -g AppleLocale 2>/dev/null || echo en_US)"
  loc="${loc%%@*}"; loc="${loc/_/-}"
  case " en-US it-IT es-ES fr-FR de-DE pt-BR pl-PL zh-CN tr-TR ru-RU uk-UA vi-VN ko-KR " in
    *" $loc "*) echo "$loc" ;;
    *) echo "en-US" ;;
  esac
}

wp_cfg() {           # wp_cfg <key> : read a field from wire-pod's live config (server.epconfig, pastinitialsetup, ...)
  curl -fsS -m 5 "http://localhost:8080/api/get_config" 2>/dev/null | "$HOME_DIR/venv/bin/python" -c \
    'import json,sys; c=json.load(sys.stdin)
for k in sys.argv[1].split("."): c=(c or {}).get(k)
print(int(bool(c)))' "$1" 2>/dev/null || echo 0
}

wp_repair_config() { # setup marked done but no connection mode: WirePod crashes looking for IP-mode certs
  [[ -f "$WP_CONF_DIR/apiConfig.json" ]] || return 1
  "$HOME_DIR/venv/bin/python" - "$WP_CONF_DIR/apiConfig.json" "$WP_CONF_DIR/certs/cert.key" <<'PY'
import json, os, sys
path, key = sys.argv[1:3]
c = json.load(open(path))
server = c.setdefault("server", {})
if c.get("pastinitialsetup") and not server.get("epconfig") and not os.path.exists(key):
    server["epconfig"], server["port"] = True, "443"     # escape-pod mode, what wire-pod's own setup picks
    json.dump(c, open(path, "w"))
    sys.exit(0)
sys.exit(1)
PY
}

wp_autosetup() {     # finish wire-pod's first-run setup so its web UI never asks: speech language + escape-pod mode
  local url="http://localhost:8080" lang i
  for i in $(seq 1 120); do curl -fsS -m 2 -o /dev/null "$url/api/is_running" 2>/dev/null && break; sleep 0.5; done
  if [[ "$(wp_cfg pastinitialsetup)" != "1" ]]; then
    lang="$(wp_lang)"
    for i in $(seq 1 3); do                   # first call may only start the language model download
      curl -fsS -m 10 -X POST -H 'Content-Type: application/json' -d "{\"language\":\"$lang\"}" \
        "$url/api/set_stt_info" 2>/dev/null | grep -q "downloading" || break
      for _ in $(seq 1 600); do               # up to ~5 min for the speech model
        case "$(curl -fsS -m 5 "$url/api/get_download_status" 2>/dev/null)" in
          *success*|*"not downloading"*) break ;;
          *error*) return 1 ;;
        esac
        sleep 0.5
      done
    done
  fi
  # the second half of wire-pod's own setup page: how robots reach it. Escape-pod mode
  # (escapepod.local, port 443) is what the robot's wire-pod firmware expects.
  if [[ "$(wp_cfg server.epconfig)" != "1" ]]; then
    curl -fsS -m 20 -o /dev/null "$url/api-chipper/use_ep" 2>/dev/null || true
  fi
}

wp_ensure() {        # installer/updater entry point: make sure a compatible wire-pod is there and running
  local pinned; pinned="$(wp_pinned "$HOME_DIR/app/compat.json")"
  if wp_managed; then
    [[ "$(wp_installed_version)" == "$pinned" ]] || wp_install "$pinned"
    if wp_repair_config; then                # fixed a half-finished setup: restart the crashed copy
      echo "repaired wire-pod's setup (escape-pod mode)"
      wp_quit
    fi
    wp_running || open -g -a "$WP_APP"
    wp_autosetup
  elif url="$(wp_reachable)"; then
    echo "using existing wire-pod at $url"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    wp_install "$pinned"
    wp_autosetup
  else
    echo "no wire-pod found; install it yourself on Linux (see README)"
  fi
}
