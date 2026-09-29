#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

# Formats written to calls-results together with <name>.json.
# Change this line (or pass a different value) to pick your outputs.
FORMATS="${TRANSCRIBER_FORMATS:-md,txt,html}"

exec transcriber watch --format "$FORMATS" "$@"