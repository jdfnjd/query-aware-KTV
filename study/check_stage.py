"""Technical gate only; no threshold or tuning based on answer correctness."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--split',required=True)
a=p.parse_args(); root=Path(a.root)
summary=json.loads((root/(a.split+'_summary.json')).read_text())
expected=len(json.loads((root/(a.split+'.json')).read_text()))
assert summary['n']==expected
assert len(summary['modes'])==4
for mode,metrics in summary['modes'].items():
    assert metrics['all']['n']==expected,mode
    assert metrics['peak_allocated_mb']['max']<32768,mode
    # A parsing/system failure should stop unattended expansion of the run.
    assert metrics['invalid_predictions']<=max(0,int(expected*.05)),(mode,'excess invalid output')
print(f'Technical gate passed: {a.split}, {expected} paired questions, four modes')
