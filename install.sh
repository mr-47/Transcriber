#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo ">> Creating virtualenv (.venv)"
python3 -m venv .venv

echo ">> Installing transcriber (and dev extras)"
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"

if [[ -z "${TRANSCRIBER_HF_TOKEN:-}" ]]; then
  echo
  echo ">> Note: TRANSCRIBER_HF_TOKEN is not set, speaker diarization is disabled."
  echo "   Export it once (or add to ~/.bashrc) to enable speaker labels:"
  echo "     export TRANSCRIBER_HF_TOKEN=hf_..."
fi

echo
echo "Install complete. Start folder monitoring with:"
echo "  ./run.sh"