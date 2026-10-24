"""Verify published predictions, budgets, split membership and timing summaries."""
import json
import math
from pathlib import Path
from statistics import mean

base=Path(__file__).resolve().parents[1]
modes=('ktv6','hybrid4_neighbor','hybrid4_top2','query6')
for split,n in [('pilot',100),('subset400',400)]:
    rows=[json.loads(line) for line in (base/'results'/(split+'_0.jsonl')).read_text().splitlines()]
    keyed={(r['id'],r['mode']):r for r in rows}
    expected=set((base/'splits'/(split+'_ids.txt')).read_text().splitlines())
    assert len(rows)==len(keyed)==n*4 and len(expected)==n
    assert all((qid,mode) in keyed for qid in expected for mode in modes)
    summary=json.loads((base/'results'/(split+'_summary.json')).read_text())
    for r in rows:
        assert r['visual_tokens']==936 and len(set(r['frame_indices']))==6
        assert r['frame_indices']==sorted(r['frame_indices'])
        assert sum(r['selection']['tokens_per_frame'])==936
        assert r['correct']==(r['predicted_answer']==r['answer'])
    for mode in modes:
        accuracy=100*mean(keyed[qid,mode]['correct'] for qid in expected)
        assert math.isclose(accuracy,summary['modes'][mode]['all']['accuracy_pct'])
        if mode!='ktv6':
            comparison=summary['comparisons'][mode]['all']
            wins=sum(not keyed[qid,'ktv6']['correct'] and keyed[qid,mode]['correct'] for qid in expected)
            losses=sum(keyed[qid,'ktv6']['correct'] and not keyed[qid,mode]['correct'] for qid in expected)
            assert (wins,losses)==(comparison['wins'],comparison['losses'])
    print(split,': paired predictions, accuracy, flips and budgets verified')
timings=[json.loads(line) for line in (base/'results/e2e_cost/measurements.jsonl').read_text().splitlines()]
summary=json.loads((base/'results/e2e_cost/summary.json').read_text())
assert len(timings)==len({(r['id'],r['mode']) for r in timings})==32
assert len({r['video'] for r in timings})==8
for mode in modes:
    rr=[r for r in timings if r['mode']==mode]
    assert len(rr)==8
    assert math.isclose(mean(r['end_to_end_s'] for r in rr),summary['modes'][mode]['mean_s'])
    assert all(r['visual_tokens']==936 and r['end_to_end_s']>r['selection_pipeline_s'] for r in rr)
print('Direct timing: 32 requests, budgets and mean latencies verified')
