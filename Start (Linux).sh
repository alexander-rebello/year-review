#!/bin/bash
# Start Year Review: double-click (choose "Run") or run ./"Start (Linux).sh" in a terminal.
cd "$(dirname "$0")/app" || exit 1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUTF8=1
if ! command -v uv >/dev/null 2>&1; then
  echo "First start: installing uv, which takes care of Python for this app (one time) ..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Could not install uv."; exit 1; }
  else
    wget -qO- https://astral.sh/uv/install.sh | sh || { echo "Could not install uv."; exit 1; }
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
exec uv run --python 3.12 review.py "$@"
