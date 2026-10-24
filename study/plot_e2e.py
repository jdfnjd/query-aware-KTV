"""Scientific plot of direct per-video measurements; no fitted trend."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[1]
rows=[json.loads(line) for line in (base/'results/e2e_cost/measurements.jsonl').read_text().splitlines()]
names={'ktv6':'KTV baseline','hybrid4_neighbor':'Hybrid + neighbor',
       'hybrid4_top2':'Hybrid without neighbor','query6':'Query-only'}
colors=['#303f60','#d65b34','#a680bf','#299179']
fig,ax=plt.subplots(figsize=(7.8,4.5),layout='constrained')
for (mode,label),color in zip(names.items(),colors):
    rr=sorted([r for r in rows if r['mode']==mode],key=lambda r:r['candidates'])
    ax.plot([r['candidates'] for r in rr],[r['end_to_end_s'] for r in rr],
            marker='o',markersize=5,linewidth=1.4,color=color,label=label)
ax.set(xlabel='Candidate frames in video',ylabel='Complete request time (seconds)',
       title='Measured VideoQA cost with features recomputed')
ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,fontsize=9)
fig.text(.5,-.035,'8 videos; one request per method/video; resident models; V100 GPU 1.',
         ha='center',fontsize=9)
out=base/'results/e2e_cost'
fig.savefig(out/'end_to_end_cost.png',dpi=180,bbox_inches='tight')
fig.savefig(out/'end_to_end_cost.svg',bbox_inches='tight')
print('Saved end_to_end_cost.png and .svg')
