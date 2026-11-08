#!/usr/bin/env bash
# CloakOS Vector Edition moved to vector-edition/. This shim keeps old one-liners and
# already-installed Macs (whose updater fetches this path) working.
set -euo pipefail
ref="${CLOAK_REF:-main}"
if [[ -n "${CLOAK_SOURCE:-}" && -f "$CLOAK_SOURCE/vector-edition/install.sh" ]]; then
  exec env CLOAK_SOURCE="$CLOAK_SOURCE/vector-edition" bash "$CLOAK_SOURCE/vector-edition/install.sh"
fi
curl -fsSL "https://raw.githubusercontent.com/UseCloakAI/CloakOS/$ref/vector-edition/install.sh" | bash
