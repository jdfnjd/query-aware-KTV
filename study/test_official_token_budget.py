"""Execute the official pruning function on synthetic features, without weights."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch


class OfficialTokenBudgetTests(unittest.TestCase):
    def test_actual_projected_length_and_rank_allocation(self):
        source=Path(os.environ.get('KTV_SOURCE_DIR',str(Path(__file__).resolve().parents[1]/'upstream/KTV-official')))/'ktv/llava/model/llava_arch.py'
        tree=ast.parse(source.read_text(encoding='utf8'))
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='encode_images')
        isolated=ast.Module(body=[fn],type_ignores=[])
        scope={'torch':torch}; exec(compile(isolated,str(source),'exec'),scope)
        torch.manual_seed(42)
        patches=torch.randn(6,577,8); cls=torch.randn(6,576)
        calls=[]
        def project(x):
            calls.append(len(x)); return x
        backend=SimpleNamespace(get_vision_tower=lambda:lambda images,mode:(patches,cls),mm_projector=project)
        dummy=SimpleNamespace(get_model=lambda:backend)
        for ranks in ([0,1,2,3,4,5],[5,4,3,2,1,0],[2,0,5,1,4,3]):
            calls.clear()
            with redirect_stdout(io.StringIO()):
                output=scope['encode_images'](dummy,None,ranks,6,'cls_new_token_sim',.2,936)
            self.assertEqual(tuple(output.shape),(1,936,8))
            self.assertEqual(calls,[[288,144,144,144,144,72][r] for r in ranks])
            self.assertTrue(torch.isfinite(output).all())

if __name__=='__main__': unittest.main()
