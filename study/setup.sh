#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$STUDY_BASE"
if [[ ! -x "$PY" ]]; then python3 -m venv "$STUDY_BASE/venv"; fi
"$PY" -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install -r "$CODE_DIR/requirements.txt"
"$PY" "$CODE_DIR/study/fetch_sources.py"
"$PY" -m unittest discover -s "$CODE_DIR/study" -p 'test_*.py'
