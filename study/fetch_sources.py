"""Fetch immutable upstream code; preserve any existing checkout."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(); p.add_argument('--related',action='store_true'); a=p.parse_args()
base=Path(__file__).resolve().parents[1]
sources=json.loads((base/'config/sources.json').read_text())
for name,entry in sources.items():
    if name not in ('KTV-official','NExT-QA') and not a.related: continue
    path=base/'upstream'/name
    if path.exists():
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=entry['commit']: raise RuntimeError(f'{name}: existing checkout has a different commit; leaving it untouched')
        print(name,actual); continue
    path.mkdir(parents=True)
    subprocess.run(['git','init',str(path)],check=True)
    subprocess.run(['git','-C',str(path),'remote','add','origin',entry['url']],check=True)
    subprocess.run(['git','-C',str(path),'fetch','--depth','1','origin',entry['commit']],check=True)
    subprocess.run(['git','-C',str(path),'checkout','--detach','FETCH_HEAD'],check=True)
    print(name,entry['commit'])
