#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a local virtual environment and install project dependencies.
# Usage:
#   scripts/bootstrap_venv.sh
# Optional env vars:
#   PYTHON_BIN=python3.11 VENV_DIR=.venv-dev scripts/bootstrap_venv.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  cat <<'EOF'
Error: python executable not found.

On Apple Silicon (including M3 Max), install Python via Homebrew:
  brew install python@3.11
Then rerun:
  PYTHON_BIN=python3.11 scripts/bootstrap_venv.sh
EOF
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "Warning: expected Apple Silicon (arm64) but got '$ARCH'. Continuing..." >&2
fi

echo "Creating virtual environment in '$VENV_DIR' using '$PYTHON_BIN'..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip tooling..."
pip install --upgrade pip setuptools wheel

echo "Installing core dependencies..."
pip install -r requirements.txt

cat <<'EOF'
Done!

Activate the environment with:
  source .venv/bin/activate

To enable voice isolation, install RNNoise (build tools required):
  pip install rnnoise

If RNNoise build fails on Apple Silicon, ensure you have Homebrew dependencies:
  brew install autoconf automake libtool pkg-config
EOF
