#!/usr/bin/env bash
CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export STUDY_BASE="${STUDY_BASE:-$CODE_DIR/.runtime}"
export HF_HOME="${HF_HOME:-$STUDY_BASE/hf}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
BASE="$STUDY_BASE"
PY="${STUDY_PYTHON:-$STUDY_BASE/venv/bin/python}"
