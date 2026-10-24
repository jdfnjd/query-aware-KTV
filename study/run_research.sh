#!/usr/bin/env bash
# Detached orchestration: no interactive assistant or SSH session is required.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -euo pipefail
DEVICE="${1:-auto}"
FINAL_STAGE="${2:-subset400}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$CODE_DIR/logs" "$CODE_DIR/results_backup"
exec 8>"$CODE_DIR/research.lock"
flock -n 8 || { echo 'Research supervisor already running'; exit 1; }
STATUS="$CODE_DIR/research-status.txt"
trap 'rc=$?; printf "FAILED exit=%s at %s; inspect logs/research.log and stage logs\n" "$rc" "$(date -Is)" > "$STATUS"' ERR
for STAGE in smoke pilot "$FINAL_STAGE"; do
    printf 'Waiting for stage lock: %s at %s\n' "$STAGE" "$(date -Is)" > "$STATUS"
    # Let an existing smoke test finish, without starting a duplicate worker.
    flock "$CODE_DIR/gpu-stage.lock" true
    if [[ -f "$STUDY_BASE/data/${STAGE}_summary.json" ]]; then
        echo "Revalidating completed stage $STAGE"
        "$PY" "$CODE_DIR/study/analyze.py" --root "$STUDY_BASE/data" --split "$STAGE" > "$CODE_DIR/logs/${STAGE}_analysis.log"
    else
        bash "$CODE_DIR/study/run_stage.sh" "$STAGE" "$DEVICE"
    fi
    "$PY" "$CODE_DIR/study/check_stage.py" --root "$STUDY_BASE/data" --split "$STAGE"
    cp -a "$STUDY_BASE/data/results" "$STUDY_BASE/data/${STAGE}_summary.json" "$CODE_DIR/results_backup/"
done
"$PY" "$CODE_DIR/study/cases.py" --root "$STUDY_BASE/data" --split "$FINAL_STAGE"
cp -a "$STUDY_BASE/data/cases" "$CODE_DIR/results_backup/"
printf 'COMPLETE: smoke, pilot, %s, summary statistics and case sheets; %s\n' "$FINAL_STAGE" "$(date -Is)" > "$STATUS"
