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
    rows=json.loads((root/(a.split+'.json')).read_text())
    videos=sorted({r['video'] for r in rows})[a.shard::a.shards]
    rows=sorted([r for r in rows if r['video'] in set(videos)],key=lambda r:(r['video'],r['id']))
    out=root/'results'; out.mkdir(exist_ok=True)
    target=out/f'{a.split}_{a.shard}.jsonl'
    existing={}
    # Full run reuses pilot predictions instead of rerunning and changing them.
    paths=list(out.glob('pilot_*.jsonl')) if a.split in ('full','subset1000','subset400') else []
    if target.exists(): paths.append(target)
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            record=json.loads(line); existing[(record['id'],record['mode'])]=record
    recorded=set()
    if target.exists():
        recorded={(json.loads(line)['id'],json.loads(line)['mode']) for line in target.read_text().splitlines() if line.strip()}
    cache_vid=None; warmed=False
    with target.open('a',buffering=1) as f:
        for ordinal,row in enumerate(rows):
            if all((row['id'],m) in existing for m in MODES):
                for mode in MODES:
                    key=(row['id'],mode)
                    if key not in recorded:
                        f.write(json.dumps(existing[key])+'\n'); recorded.add(key)
                continue
            if cache_vid!=row['video']:
                feature=np.load(root/'features'/(row['video']+'.npz'))
                meta=json.loads(str(feature['metadata'])); cache_vid=row['video']
            torch.cuda.synchronize(); start=time.perf_counter()
            # Match official long-question truncation, including its word window.
            words=row['question'].split(' ')
            clip_query=' '.join(words[10:50]) if len(words)>40 else row['question']
            text=proc(text=[clip_query],return_tensors='pt',padding=True,truncation=True)
            with torch.inference_mode():
                embedding=clip.get_text_features(**{k:v.cuda() for k,v in text.items()})
                embedding=embedding/embedding.norm(dim=-1,keepdim=True)
                scores=(torch.tensor(feature['clip'],device='cuda',dtype=torch.float16) @ embedding[0]).float().cpu().numpy()
            torch.cuda.synchronize(); query_s=time.perf_counter()-start
            # Rotate mode order to reduce systematic warm-cache/temperature effects.
            rotation=ordinal%4; modes=MODES[rotation:]+MODES[:rotation]
            for mode in modes:
                key=(row['id'],mode)
                if key in existing:
                    if key not in recorded: f.write(json.dumps(existing[key])+'\n'); recorded.add(key)
                    continue
                start=time.perf_counter()
                selected=select(mode,feature['base4'].tolist(),feature['base6'].tolist(),scores,feature['times'],row['question'])
                selection_s=time.perf_counter()-start
                positions=selected['indices']; raw_indices=feature['indices'][positions].tolist()
                start=time.perf_counter()
                frames=load_frames(root/'videos'/(row['video']+'.mp4'),raw_indices)
                decode_s=time.perf_counter()-start
                prompt=get_multiple_choice_prompt(model,'multiple_choice_allvideo_v4',row['question'],row['candidates'])
                def generate():
                    ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
                    images=process_images(frames,image_processor,model.config).cuda().half()
                    with torch.inference_mode(),redirect_stdout(io.StringIO()):
                        output=model.generate(ids,images=images,image_sizes=tuple(im.size for im in frames),
                            do_sample=False,num_beams=1,max_new_tokens=128,use_cache=True,
                            keyframe_order=selected['ranks'],num_frames=6,
                            prune_mode='cls_new_token_sim',global_rate=.2,tokens_num=936)
                    return tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip()
                if not warmed:
                    generate(); torch.cuda.synchronize(); warmed=True
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                prediction=generate(); torch.cuda.synchronize(); inference_s=time.perf_counter()-start
                answer=parse_answer(prediction)
                record={**row,'mode':mode,'prediction':prediction,'predicted_answer':answer,
                    'correct':answer==row['answer'],'invalid_prediction':answer<0,
                    'frame_indices':raw_indices,'frame_times':feature['times'][positions].tolist(),
                    'clip_scores':scores[positions].tolist(),'selection':selected,
                    'visual_tokens':model.study_visual_tokens,'query_s':query_s,
                    'selection_s':selection_s,'selected_decode_s':decode_s,'inference_s':inference_s,
                    'cached_wall_s':query_s+selection_s+decode_s+inference_s,
                    'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
                    'peak_reserved_mb':torch.cuda.max_memory_reserved()/1024**2,
                    'feature_metrics':meta,'device':torch.cuda.get_device_name(),
                    'physical_gpu':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                    'clip_query':clip_query,'split_first_run':a.split}
                f.write(json.dumps(record)+'\n'); existing[key]=record; recorded.add(key)
                print(json.dumps({'progress':f'{ordinal+1}/{len(rows)}','id':row['id'],'mode':mode,
                                  'correct':record['correct'],'inference_s':inference_s}),flush=True)

if __name__=='__main__': main()
