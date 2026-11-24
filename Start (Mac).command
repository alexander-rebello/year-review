#!/bin/bash
# Double-click to start Year Review. The first start sets everything up by itself.
cd "$(dirname "$0")/app" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"   # Homebrew is not on PATH when double-clicked
export PYTHONUTF8=1
if ! command -v uv >/dev/null 2>&1; then
  echo "First start: installing uv, which takes care of Python for this app (one time) ..."
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    echo; read -r -p "Could not install uv. Are you online? Press Enter to close."; exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
uv run --python 3.12 review.py "$@" || { echo; read -r -p "Something went wrong (see above). Press Enter to close."; }
