#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
if [[ -z "${STUDY_QUOTA_FILESYSTEM:-}" && -z "${STUDY_STORAGE_BUDGET_GIB:-}" ]]; then
    echo 'Check quota -s -v and permissions; set STUDY_QUOTA_FILESYSTEM or STUDY_STORAGE_BUDGET_GIB.' >&2
    exit 2
fi
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180 HF_HUB_ETAG_TIMEOUT=60
mkdir -p "$STUDY_BASE/data"
if [[ ! -f "$STUDY_BASE/data/asset_pins.json" ]]; then
    cp "$CODE_DIR/config/model_revisions.json" "$STUDY_BASE/data/asset_pins.json"
fi
"$PY" "$CODE_DIR/study/prepare.py" --root "$STUDY_BASE/data" \
    --annotations "$CODE_DIR/upstream/NExT-QA/dataset/nextqa/val.csv" \
    --mapping "$CODE_DIR/upstream/NExT-QA/dataset/nextqa/map_vid_vidorID.json"
"$PY" "$CODE_DIR/study/prepare_subset400.py" --root "$STUDY_BASE/data"
"$PY" - "$STUDY_BASE/data" "$CODE_DIR" <<'PY'
import json,sys
from pathlib import Path
root,code=map(Path,sys.argv[1:])
for split in ('pilot','subset400'):
    actual=sorted(r['id'] for r in json.loads((root/(split+'.json')).read_text()))
    expected=(code/'splits'/(split+'_ids.txt')).read_text().splitlines()
    assert actual==expected, f'{split}: membership differs from the published run'
print('Frozen pilot and 400-question split match the published IDs.')
PY
