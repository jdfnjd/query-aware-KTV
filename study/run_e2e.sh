#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HOME="$BASE/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec 9>"$CODE_DIR/gpu-stage.lock"
flock -n 9 || exit 4
export GPU_STATUS_FILE="$CODE_DIR/e2e-status.txt"
export CUDA_VISIBLE_DEVICES=$(bash "$CODE_DIR/study/wait_gpu.sh" auto 'end-to-end benchmark')
trap 'printf "FAILED end-to-end benchmark; inspect logs/e2e.log\n" > "$CODE_DIR/e2e-status.txt"' ERR
echo "RUNNING on GPU $CUDA_VISIBLE_DEVICES" > "$CODE_DIR/e2e-status.txt"
"$BASE/venv/bin/python" -u "$CODE_DIR/study/benchmark_e2e.py" --root "$BASE/data" --upstream "$CODE_DIR/upstream/KTV-official"
cp -a "$BASE/data/e2e_cost" "$CODE_DIR/results_backup/"
echo COMPLETE > "$CODE_DIR/e2e-status.txt"
