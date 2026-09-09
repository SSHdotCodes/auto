#!/bin/sh
set -eu
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'Installing uv from astral.sh to manage an isolated Python runtime…'
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi
uv python install 3.13
AUTO_PYTHON="$(uv python find 3.13)"
AUTO_SCRIPT="$(mktemp)"
trap 'rm -f "$AUTO_SCRIPT"' EXIT HUP INT TERM
curl -fsSL https://auto.ssh.codes/install.py -o "$AUTO_SCRIPT"
"$AUTO_PYTHON" "$AUTO_SCRIPT" "$@"
