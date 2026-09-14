#!/usr/bin/env bash
# Start the API and UI on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/.."
exec uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
