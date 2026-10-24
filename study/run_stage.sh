#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HOME="$STUDY_BASE/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
STAGE="${1:-smoke}"
GPU_REQUEST="${2:-auto}"
case "$STAGE" in smoke|pilot|full|subset1000|subset400) ;; *) echo 'Unknown study split'; exit 2;; esac
[[ "$GPU_REQUEST" == auto || "$GPU_REQUEST" =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$CODE_DIR/logs" "$STUDY_BASE/data"
exec 9>"$CODE_DIR/gpu-stage.lock"
flock -n 9 || { echo 'Another study GPU stage is active'; exit 4; }
if command -v quota >/dev/null; then quota -s -v || true; fi
DEVICE=$(bash "$CODE_DIR/study/wait_gpu.sh" "$GPU_REQUEST" "$STAGE features")
export CUDA_VISIBLE_DEVICES="$DEVICE"
"$PY" -u "$CODE_DIR/study/features.py" --root "$STUDY_BASE/data" --split "$STAGE" --shard 0 --shards 1 > "$CODE_DIR/logs/${STAGE}_features_0.log" 2>&1
DEVICE=$(bash "$CODE_DIR/study/wait_gpu.sh" "$GPU_REQUEST" "$STAGE inference")
export CUDA_VISIBLE_DEVICES="$DEVICE"
printf 'RUNNING %s inference on GPU %s, %s\n' "$STAGE" "$DEVICE" "$(date -Is)" > "$CODE_DIR/research-status.txt"
"$PY" -u "$CODE_DIR/study/infer.py" --root "$STUDY_BASE/data" --upstream "$CODE_DIR/upstream/KTV-official" --split "$STAGE" --shard 0 --shards 1 > "$CODE_DIR/logs/${STAGE}_infer_0.log" 2>&1
"$PY" "$CODE_DIR/study/analyze.py" --root "$STUDY_BASE/data" --split "$STAGE" > "$CODE_DIR/logs/${STAGE}_analysis.log"
mkdir -p "$CODE_DIR/results_backup"
cp -a "$STUDY_BASE/data/results" "$STUDY_BASE/data/${STAGE}_summary.json" "$CODE_DIR/results_backup/"
if command -v quota >/dev/null; then quota -s -v || true; fi
echo "Stage $STAGE finished on GPU $DEVICE"
