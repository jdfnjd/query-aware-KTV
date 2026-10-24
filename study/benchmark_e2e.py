"""Run paired modes through the unmodified official KTV model implementation."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import time
import types
import av
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from selection import MODES, select


def load_frames(path, indices):
    wanted=set(indices); frames=[]
    with av.open(str(path)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in wanted: frames.append(Image.fromarray(frame.to_ndarray(format='rgb24')))
            if i>=max(indices): break
    if len(frames)!=6: raise ValueError('Did not decode exactly six frames')
    return frames


def parse_answer(text):
    # Prompt already ends with "Best option:(". Accept only an initial option;
    # invalid output is a wrong answer, never repaired using the ground truth.
    match=re.match(r'^\s*\(?([A-E])(?:\b|\))',text)
    if match: return ord(match.group(1))-ord('A')
    return -1


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    p.add_argument('--upstream',required=True); p.add_argument('--split',default='pilot')
    p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=1)
    a=p.parse_args(); root=Path(a.root); upstream=Path(a.upstream).resolve()
    sys.path[:0]=[str(upstream/'ktv'),str(upstream)]
    from llava.model.language_model.llava_llama import LlavaConfig,LlavaLlamaForCausalLM
    from transformers import AutoTokenizer
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images,tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    from prompt import get_multiple_choice_prompt
    torch.set_num_threads(4); torch.manual_seed(20260927); np.random.seed(20260927)
    pins=json.loads((root/'asset_pins.json').read_text())
    setup_start=time.perf_counter()
    disable_torch_init()
    model_path=pins['liuhaotian/llava-v1.6-vicuna-7b']['path']
    tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=False)
    config=LlavaConfig.from_pretrained(model_path)
    # Resolve the vision tower to the pinned local snapshot rather than letting
    # the original builder resolve its model ID against a changing main branch.
    config.mm_vision_tower=pins['openai/clip-vit-large-patch14-336']['path']
    config.rope_scaling={'factor':2.,'type':'dynamic'}
    config.max_sequence_length=8192; config.tokenizer_model_max_length=8192
    model,loading_info=LlavaLlamaForCausalLM.from_pretrained(model_path,config=config,
        low_cpu_mem_usage=True,device_map='cuda:0',torch_dtype=torch.float16,output_loading_info=True)
    if loading_info.get('missing_keys') or loading_info.get('mismatched_keys') or loading_info.get('error_msgs'):
        raise RuntimeError(f'Incomplete checkpoint loading: {loading_info}')
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError('Model contains unloaded meta parameters')
    print(json.dumps({'checkpoint_loading_verified':True,
                      'unexpected_keys':loading_info.get('unexpected_keys',[]),
                      'vision_parameter_tensors':sum('vision_tower' in name for name,_ in model.named_parameters())}),flush=True)
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN,DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN
    if getattr(config,'mm_use_im_patch_token',True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN],special_tokens=True)
    if getattr(config,'mm_use_im_start_end',False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN],special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    tower=model.get_vision_tower()
    if not tower.is_loaded: tower.load_model(device_map='cuda:0')
    tower.to(device='cuda:0',dtype=torch.float16)
    image_processor=tower.image_processor
    model.eval(); model.config.image_aspect_ratio='resize'
    clip=CLIPModel.from_pretrained(pins['openai/clip-vit-large-patch14']['path']).eval().cuda().half()
    proc=CLIPProcessor.from_pretrained(pins['openai/clip-vit-large-patch14']['path'])
    # Assert actual projected-token count, not just the requested budget.
    original_encode=model.encode_images
    def checked_encode(self,*args,**kwargs):
        features=original_encode(*args,**kwargs)
        if features.shape[1]!=936 or not torch.isfinite(features).all():
            raise ValueError(f'Invalid visual token representation {features.shape}')
        self.study_visual_tokens=int(features.shape[1])
        return features
    model.encode_images=types.MethodType(checked_encode,model)
    model_setup_s=time.perf_counter()-setup_start
    from e2e_cost import run
    run(locals())

if __name__=='__main__': main()
