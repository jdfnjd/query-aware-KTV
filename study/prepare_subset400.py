"""Freeze a subtype-stratified 400-question subset, including the original pilot.

Selection never reads predictions, correctness, or feature-cache availability.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
root=Path(p.parse_args().root)
full=json.loads((root/'full.json').read_text())
pilot=json.loads((root/'pilot.json').read_text())
pilot_ids={r['id'] for r in pilot}
assert len(pilot_ids)==100 and len({r['id'] for r in full})==len(full)
counts=Counter(r['type'] for r in full)
ideal={t:400*n/len(full) for t,n in counts.items()}
quotas={t:int(n) for t,n in ideal.items()}
for t in sorted(ideal,key=lambda t:(-(ideal[t]-quotas[t]),t))[:400-sum(quotas.values())]:
    quotas[t]+=1
groups=defaultdict(list)
for r in full:
    if r['id'] not in pilot_ids: groups[r['type']].append(r)
pilot_counts=Counter(r['type'] for r in pilot)
rng=random.Random(20260928); selected=list(pilot)
for t in sorted(quotas):
    pool=sorted(groups[t],key=lambda r:r['id']); rng.shuffle(pool)
    needed=quotas[t]-pilot_counts[t]
    assert 0<=needed<=len(pool)
    selected.extend(pool[:needed])
selected.sort(key=lambda r:r['id'])
assert len(selected)==len({r['id'] for r in selected})==400
assert pilot_ids <= {r['id'] for r in selected}
target=root/'subset400.json'
payload=json.dumps(selected,indent=2)
if target.exists(): assert target.read_text()==payload, 'Refusing to replace a different frozen subset'
else: target.write_text(payload)
manifest={'questions':400,'videos':len({r['video'] for r in selected}),
          'pilot_included':100,'new_questions':300,'seed':20260928,'type_counts':quotas,
          'sha256':hashlib.sha256(payload.encode()).hexdigest(),
          'selection':'proportional subtype quotas including frozen pilot; predictions never read'}
(root/'subset400_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))

