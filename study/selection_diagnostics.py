"""Post-hoc descriptive diagnostics; does not alter any policy or prediction."""
import json
from pathlib import Path
from statistics import mean, median

folder=Path(__file__).resolve().parents[1]/'results'
rows=[json.loads(line) for line in (folder/'subset400_0.jsonl').read_text().splitlines()]
modes=sorted({r['mode'] for r in rows}); keyed={(r['id'],r['mode']):r for r in rows}
assert len(rows)==len(keyed)==1600
out={}
for mode in modes:
    rr=[r for r in rows if r['mode']==mode]
    gaps=[b-a for r in rr for a,b in zip(r['frame_times'],r['frame_times'][1:])]
    out[mode]={'questions':len(rr),'mean_span_s':mean(r['frame_times'][-1]-r['frame_times'][0] for r in rr),
               'median_adjacent_gap_s':median(gaps),
               'adjacent_pairs_under_0_1s':sum(g<.1 for g in gaps),'adjacent_pairs_total':len(gaps)}
    if mode=='hybrid4_neighbor':
        out[mode]['anchor_already_in_context']=sum(r['selection']['anchor_in_base4'] for r in rr)
        out[mode]['vs_no_neighbor']={
            'wins':sum(r['correct'] and not keyed[r['id'],'hybrid4_top2']['correct'] for r in rr),
            'losses':sum(not r['correct'] and keyed[r['id'],'hybrid4_top2']['correct'] for r in rr)}
(folder/'selection_diagnostics.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
