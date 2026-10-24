"""Paired question accuracy with video-cluster bootstrap uncertainty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from selection import MODES


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--split',default='pilot')
    a=p.parse_args(); root=Path(a.root); rows={}
    expected=json.loads((root/(a.split+'.json')).read_text()); ids={r['id'] for r in expected}
    for path in sorted((root/'results').glob(a.split+'_*.jsonl')):
        for line in path.read_text().splitlines():
            r=json.loads(line); key=(r['id'],r['mode'])
            if key in rows: raise ValueError(f'Duplicate prediction {key}')
            rows[key]=r
    complete=[i for i in sorted(ids) if all((i,m) in rows for m in MODES)]
    if len(complete)!=len(ids): raise ValueError(f'Incomplete paired run: {len(complete)}/{len(ids)}')
    report={'split':a.split,'n':len(complete),'modes':{},'comparisons':{},'cases':{}}
    for mode in MODES:
        rr=[rows[i,mode] for i in complete]
        if any(r['visual_tokens']!=936 or len(set(r['frame_indices']))!=6 for r in rr):
            raise ValueError('Budget mismatch')
        groups=defaultdict(list)
        for r in rr:
            groups['all'].append(r)
            groups[r['type'][0]].append(r)
            groups['TN' if r['type']=='TP' else r['type']].append(r)
        result={g:{'n':len(rs),'accuracy_pct':100*np.mean([r['correct'] for r in rs])} for g,rs in groups.items()}
        for metric in ['inference_s','cached_wall_s','peak_allocated_mb','peak_reserved_mb','selection_s','query_s']:
            values=[r[metric] for r in rr]
            result[metric]={'mean':float(np.mean(values)),'median':float(np.median(values)),
                            'p95':float(np.percentile(values,95)),'max':float(max(values))}
        result['invalid_predictions']=sum(r['invalid_prediction'] for r in rr)
        # Cold-path components cannot be called measured end-to-end wall time.
        cold=[]
        for r in rr:
            fm=r['feature_metrics']
            base=fm['decode_s']
            if mode=='ktv6':
                base+=fm['dino_s']+fm['kmeans6_s']+fm['baseline_clip_decode_s']+fm['baseline_clip_s']
            elif mode=='query6':
                base+=fm['clip_image_s']
            else:
                base+=fm['dino_s']+fm['kmeans4_s']+fm['clip_image_s']
            cold.append(base+r['cached_wall_s'])
        result['reconstructed_cold_path_s_mean']=float(np.mean(cold))
        report['modes'][mode]=result
    for mode in MODES[1:]:
        base=[rows[i,'ktv6'] for i in complete]; other=[rows[i,mode] for i in complete]
        comparisons={}
        for group in ('all','C','T','D'):
            pairs=[(x,y) for x,y in zip(base,other) if group=='all' or x['type'].startswith(group)]
            if not pairs: continue
            by_video=defaultdict(list)
            for x,y in pairs: by_video[x['video']].append(int(y['correct'])-int(x['correct']))
            totals=np.array([sum(v) for v in by_video.values()]); counts=np.array([len(v) for v in by_video.values()])
            rng=np.random.default_rng(20260927); boot=[]
            for _ in range(10000):
                ix=rng.integers(0,len(totals),len(totals)); boot.append(100*totals[ix].sum()/counts[ix].sum())
            wins=sum(not x['correct'] and y['correct'] for x,y in pairs)
            losses=sum(x['correct'] and not y['correct'] for x,y in pairs)
            comparisons[group]={'n':len(pairs),'videos':len(by_video),'wins':wins,'losses':losses,
                'delta_pp':100*(wins-losses)/len(pairs),
                'video_bootstrap_95ci_pp':np.percentile(boot,[2.5,97.5]).tolist(),
                'mcnemar_exact_p_descriptive':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.0}
        report['comparisons'][mode]=comparisons
        report['cases'][mode]={
            'improvements':[i for i in complete if not rows[i,'ktv6']['correct'] and rows[i,mode]['correct']],
            'deteriorations':[i for i in complete if rows[i,'ktv6']['correct'] and not rows[i,mode]['correct']]}
    if a.split in ('full','subset1000','subset400'):
        pilot={r['id'] for r in json.loads((root/'pilot.json').read_text())}
        report['nonpilot_accuracy_pct']={m:100*np.mean([rows[i,m]['correct'] for i in complete if i not in pilot]) for m in MODES}
    (root/(a.split+'_summary.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__': main()
