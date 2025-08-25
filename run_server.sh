#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Install/upgrade dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Load environment variables from config.env if present
if [ -f "$SCRIPT_DIR/config.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/config.env"
  set +a
fi

# Default port if not set
export PORT="${PORT:-51174}"

# Run the FastAPI server
exec python "$SCRIPT_DIR/server.py" 