#!/usr/bin/env sh
# Always uses repo .venv so pyenv / PATH quirks cannot pick the wrong interpreter.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec "${ROOT}/.venv/bin/python" -m src.webapp
