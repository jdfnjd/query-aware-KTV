#!/usr/bin/env bash
set -euo pipefail
REQUEST="${1:-auto}"
PHASE="${2:-study}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
STATUS="${GPU_STATUS_FILE:-$CODE_DIR/research-status.txt}"
while true; do
    if [[ "$REQUEST" == auto ]]; then
        INVENTORY=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    else
        INVENTORY=$(nvidia-smi -i "$REQUEST" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    fi
    while IFS=, read -r INDEX USED UTIL; do
        INDEX=${INDEX// /}; USED=${USED// /}; UTIL=${UTIL// /}
        if [[ "$INDEX" =~ ^[0-9]+$ && "$USED" =~ ^[0-9]+$ && "$UTIL" =~ ^[0-9]+$ ]] && (( USED < 500 && UTIL < 5 )); then
            printf 'RUNNING %s on GPU %s, %s\n' "$PHASE" "$INDEX" "$(date -Is)" > "$STATUS"
            echo "$INDEX"
            exit 0
        fi
    done <<< "$INVENTORY"
    printf 'Waiting for available GPU (%s), %s, %s\n' "$REQUEST" "$PHASE" "$(date -Is)" > "$STATUS"
    sleep 30
done
