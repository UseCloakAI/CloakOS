#!/usr/bin/env bash
# Rebuild cloak_vector/web/static/pair/rts.js from pair/src (needs node + npx).
set -euo pipefail
cd "$(dirname "$0")/.."
npx --yes browserify@17.0.0 pair/src/main.js -o cloak_vector/web/static/pair/rts.js
echo "built cloak_vector/web/static/pair/rts.js"
