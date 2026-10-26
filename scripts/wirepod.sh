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
