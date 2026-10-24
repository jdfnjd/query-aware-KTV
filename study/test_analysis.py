"""Check pairing and accounting on explicitly synthetic, disposable fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from selection import MODES


class AnalysisTests(unittest.TestCase):
    def test_known_flips_and_missing_mode(self):
        with tempfile.TemporaryDirectory(prefix='ktv_synthetic_') as folder:
            root=Path(folder); (root/'results').mkdir()
            expected=[{'id':str(i)} for i in range(4)]
            (root/'pilot.json').write_text(json.dumps(expected))
            records=[]
            for mode in MODES:
                correct=[True,False,False,True] if mode=='ktv6' else [False,True,True,True]
                for i in range(4):
                    records.append({'id':str(i),'mode':mode,'video':str(i//2),'type':'CW',
                        'correct':correct[i],'visual_tokens':936,'frame_indices':list(range(6)),
                        'invalid_prediction':False,'feature_metrics':dict(decode_s=1.,clip_image_s=2.,
                        dino_s=3.,kmeans4_s=.1,kmeans6_s=.2,baseline_clip_s=.3,baseline_clip_decode_s=.4),
                        **{m:1. for m in ['inference_s','cached_wall_s','peak_allocated_mb',
                                         'peak_reserved_mb','selection_s','query_s']}})
            path=root/'results'/'pilot_0.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            command=[sys.executable,str(Path(__file__).with_name('analyze.py')),'--root',str(root)]
            subprocess.run(command,check=True,capture_output=True)
            result=json.loads((root/'pilot_summary.json').read_text())
            self.assertEqual(result['modes']['ktv6']['all']['accuracy_pct'],50.)
            for mode in MODES[1:]:
                comparison=result['comparisons'][mode]['all']
                self.assertEqual((comparison['wins'],comparison['losses'],comparison['delta_pp']),(2,1,25.))
            path.write_text(''.join(json.dumps(r)+'\n' for r in records[:-1]))
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)

if __name__=='__main__': unittest.main()
