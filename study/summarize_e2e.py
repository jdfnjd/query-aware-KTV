"""Produce an auditable report from direct measurements, without extrapolation."""
import json
from pathlib import Path
from statistics import mean,median

base=Path(__file__).resolve().parents[1]; folder=base/'results'/'e2e_cost'
rows=[json.loads(line) for line in (folder/'measurements.jsonl').read_text().splitlines()]
manifest=json.loads((folder/'manifest.json').read_text())
original={(r['id'],r['mode']):r for r in (json.loads(line) for line in (base/'results/subset400_0.jsonl').read_text().splitlines())}
names={'ktv6':'KTV baseline','hybrid4_neighbor':'Hybrid + neighbor','hybrid4_top2':'Hybrid without neighbor','query6':'Six query-aware frames'}
keyed={(r['id'],r['mode']):r for r in rows}
assert len(rows)==len(keyed)==32
assert all((qid,mode) in keyed for qid in manifest['ids'] for mode in names)
assert all(r['visual_tokens']==936 and len(set(r['frame_indices']))==6 for r in rows)
baseline=mean(r['end_to_end_s'] for r in rows if r['mode']=='ktv6')
lines=['# Direct end-to-end VideoQA cost','',
       '## Measurement scope','',
       'Eight videos were selected at the midpoint of eight quantiles of candidate-frame count from the 291-video evaluation set. One question per video was chosen by sorted ID, independently of answer correctness. All four policies were measured once per video on the same GPU, with rotated policy order: 32 complete requests. This is a small diagnostic sample, not a runtime estimate for all 400 questions.','',
       'Each synchronized wall-clock measurement starts before opening the raw video and ends after answer generation and parsing. It includes actual frame-count validation, video decoding, image preprocessing, required feature extraction, clustering where applicable, CLIP query encoding/ranking, selected-frame decoding, token pruning, prompt processing and generation. Features are recomputed for every request and never loaded from the feature cache.','',
       'KTV computes DINO on candidates and CLIP on only its six representatives. Both hybrids compute DINO and CLIP on candidates. Query-only computes CLIP on candidates and skips DINO and clustering.','',
       'Models are already resident, and one separate video warms the networks before timing. OS/NFS file caches are not flushed. Model initialization, downloads, queue waiting and output-file logging are excluded from request latency. Thus these are **uncached-feature end-to-end requests**, not cold-machine or fresh-process startup timings.','',
       '## Measured latency','',
       '| Method | Mean request | Median request | Range | Mean / KTV |',
       '|---|---:|---:|---:|---:|']
for mode,label in names.items():
    rr=[r['end_to_end_s'] for r in rows if r['mode']==mode]
    lines.append(f'| {label} | {mean(rr):.2f} s | {median(rr):.2f} s | {min(rr):.2f}–{max(rr):.2f} s | {mean(rr)/baseline:.2f}× |')
lines+=['','### Every measured video','',
        '| Question ID | Candidate frames | KTV | Hybrid + neighbor | Hybrid without neighbor | Query-only |',
        '|---|---:|---:|---:|---:|---:|']
for qid in manifest['ids']:
    rr=[keyed[qid,m] for m in names]
    lines.append('| '+qid+' | '+str(rr[0]['candidates'])+' | '+' | '.join(f"{r['end_to_end_s']:.2f} s" for r in rr)+' |')
lines+=['','## Validity and resource accounting','',
        '- Every measured request retained six distinct frames and 936 projected visual tokens.',
        '- Physical GPUs recorded: '+', '.join(sorted({str(r['physical_gpu']) for r in rows}))+'.',
        f"- Timed LLaVA/CLIP initialization: {manifest['model_setup_s']:.2f} s; additional DINO/setup work: {manifest['additional_dino_setup_s']:.2f} s. These are separate initialization measurements, not fresh-process per-method benchmarks.",
        f"- Maximum allocated GPU memory: {max(r['peak_allocated_mib'] for r in rows)/1024:.2f} GiB. All policies share the same resident LLaVA+CLIP+DINO pool; this is not minimum standalone memory for query-only or any other policy."]
for key,label in [('frames_match_original','selected frame sets'),('ranks_match_original','token relevance ranks'),('answer_matches_original','parsed answers')]:
    good=sum(r[key] for r in rows)
    lines.append(f'- Parity with the 400-question run: {good}/32 matching {label}.')
    if good<32: lines.append('  Differences: '+', '.join(r['id']+'/'+r['mode'] for r in rows if not r[key])+'.')
budget=[288,144,144,144,144,72]
allocation_matches=sum([budget[i] for i in r['ranks']]==original[r['id'],r['mode']]['selection']['tokens_per_frame'] for r in rows)
lines.append(f'- Per-frame token allocations match in {allocation_matches}/32 requests. One rank difference only swaps frames retaining the same 144-token budget; the other swaps a 72-token and a 144-token baseline allocation. Total tokens remain 936 and parsed answers are unchanged.')
lines+=['','## Limits','',
        'One observation per method/video does not quantify repeat-run jitter. Other GPUs on the shared server were busy; CPU, filesystem and power contention were not controlled. A CPU-only case-strip generation job overlapped part of the first-video timing period, so those measurements may contain additional host contention. These results measure this implementation and this sample, rather than an optimized deployment or general speed ranking.','',
        'The baseline uses a native six-image CLIP batch instead of dense cached CLIP embeddings. Numerical batch differences can change ties, so frame/rank/answer parity is audited above. The original accuracy results are not replaced by these timing runs.','',
        '## Interpretation','']
hybrid=mean(r['end_to_end_s'] for r in rows if r['mode']=='hybrid4_neighbor')
lines.append(f'The neighbor hybrid costs {hybrid/baseline:.2f}× the baseline mean request time on these eight videos ({hybrid-baseline:+.2f} seconds). The unchanged six-frame/936-token budget constrains the language-model input, but does not remove dense CLIP selection overhead. Use these directly measured times instead of treating the earlier reconstructed component sums as measured end-to-end latency.')
(base/'results'/'end_to_end_cost.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Saved results/end_to_end_cost.txt; validated 32 paired measurements.')
