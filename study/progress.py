"""Read-only progress, counting a question complete only with all four modes."""
import json
import os
from pathlib import Path

root=Path(os.environ.get('STUDY_BASE', str(Path(__file__).resolve().parents[1]/'.runtime')))/'data'
code=Path(__file__).resolve().parents[1]
rows=json.loads((root/'subset400.json').read_text())
ids={r['id'] for r in rows}
videos={r['video'] for r in rows}
modes={'ktv6','hybrid4_neighbor','hybrid4_top2','query6'}
done={qid:set() for qid in ids}
for pattern in ('pilot_*.jsonl','subset400_*.jsonl'):
    for path in (root/'results').glob(pattern):
        for line in path.read_text().splitlines():
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                continue  # A live writer may not have finished its final line.
            if record['id'] in done and record['mode'] in modes:
                done[record['id']].add(record['mode'])
complete=sum(value==modes for value in done.values())
predictions=sum(len(value) for value in done.values())
features=sum((root/'features'/(video+'.npz')).exists() for video in videos)
print(f'Questions completed: {complete}/{len(ids)} (all four methods; includes pilot)')
print(f'Method predictions:  {predictions}/{len(ids)*len(modes)}')
print(f'Videos preprocessed: {features}/{len(videos)}')
status=code/'research-status.txt'
if status.exists(): print('Status: '+status.read_text().strip())
print('Question progress advances after feature extraction. COMPLETE also confirms analysis finished.')
cost=root/'e2e_cost'/'measurements.jsonl'
if cost.exists():
    cost_keys=set()
    for line in cost.read_text().splitlines():
        try:
            record=json.loads(line); cost_keys.add((record['id'],record['mode']))
        except json.JSONDecodeError:
            continue
    print(f'Additional end-to-end measurements: {len(cost_keys)}/32')
    extra_status=code/'e2e-status.txt'
    if extra_status.exists(): print('Benchmark: '+extra_status.read_text().strip())
