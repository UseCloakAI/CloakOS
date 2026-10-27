#!/usr/bin/env bash
# Rebuild cloakos/web/static/pair/rts.js from pair/src (needs node + npx).
set -euo pipefail
cd "$(dirname "$0")/.."
npx --yes browserify@17.0.0 pair/src/main.js -o cloakos/web/static/pair/rts.js
echo "built cloakos/web/static/pair/rts.js"
