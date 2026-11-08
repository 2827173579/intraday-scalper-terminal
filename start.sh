#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
exec python -m intraday_scalper.web.terminal_server --host "${TERMINAL_HOST:-0.0.0.0}" --port "${PORT:-8080}"
