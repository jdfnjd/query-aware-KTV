"""Direct uncached request timing; called with the unchanged inference setup."""
import json
import io
from contextlib import redirect_stdout
import os
import time
from pathlib import Path
import av
import cv2
import numpy as np
import torch
from torchvision import transforms
from transformers import Dinov2Model
from PIL import Image
from selection import MODES, ktv_frames, select
from infer import load_frames, parse_answer


def run(ctx):
    root=ctx['root']; pins=ctx['pins']; clip=ctx['clip']; proc=ctx['proc']
    model=ctx['model']; tokenizer=ctx['tokenizer']
    setup_start=time.perf_counter()
    dino=Dinov2Model.from_pretrained(pins['facebook/dinov2-large']['path']).eval().cuda()
    transform=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    references={}
    for line in (root/'results'/'subset400_0.jsonl').read_text().splitlines():
        r=json.loads(line); references[r['id'],r['mode']]=r
    # One representative question/video; select at eight candidate-count quantiles.
    by_video={}
    for r in sorted(references.values(),key=lambda r:r['id']):
        if r['mode']=='ktv6': by_video.setdefault(r['video'],r)
    ordered=sorted(by_video.values(),key=lambda r:(r['feature_metrics']['candidates'],r['video']))
    sample=[ordered[min(len(ordered)-1,int((i+.5)*len(ordered)/8))] for i in range(8)]
    out=root/'e2e_cost'; out.mkdir(exist_ok=True)
    manifest={'ids':[r['id'] for r in sample], 'sampling':'eight midpoint quantiles of candidate count, one lexicographically first question per video',
        'model_setup_s':ctx['model_setup_s'],'additional_dino_setup_s':time.perf_counter()-setup_start,
        'scope':'resident models; no feature reuse; includes actual-frame count, decoding, required encoders, clustering, selection, prompt, generation and answer parsing',
        'exclusions':'model loading, dataset download, result logging; OS file cache is not flushed',
        'memory':'common resident LLaVA+CLIP+DINO pool for all modes, not minimum per-mode deployment memory'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

    def clip_images(frames):
        x=proc(images=frames,return_tensors='pt')['pixel_values'].cuda().half()
        z=clip.get_image_features(pixel_values=x)
        return (z/z.norm(dim=-1,keepdim=True)).float().cpu().numpy()

    def answer(row,frames,ranks):
        prompt=ctx['get_multiple_choice_prompt'](model,'multiple_choice_allvideo_v4',row['question'],row['candidates'])
        ids=ctx['tokenizer_image_token'](prompt,tokenizer,ctx['IMAGE_TOKEN_INDEX'],return_tensors='pt').unsqueeze(0).cuda()
        images=ctx['process_images'](frames,ctx['image_processor'],model.config).cuda().half()
        with redirect_stdout(io.StringIO()):
            output=model.generate(ids,images=images,image_sizes=tuple(im.size for im in frames),do_sample=False,
                num_beams=1,max_new_tokens=128,use_cache=True,keyframe_order=ranks,num_frames=6,
                prune_mode='cls_new_token_sim',global_rate=.2,tokens_num=936)
        return tokenizer.batch_decode(output,skip_special_tokens=True)[0].strip()

    # Warm the networks on a separate fixed video; no reusable sample features.
    warm=json.loads((root/'smoke.json').read_text())[0]
    warmref=references.get((warm['id'],'ktv6'))
    frames=load_frames(root/'videos'/(warm['video']+'.mp4'),warmref['frame_indices'])
    with torch.inference_mode():
        dino(pixel_values=torch.stack([transform(im) for im in frames]).cuda())
        clip_images(frames)
        answer(warm,frames,warmref['selection']['ranks'])
    del frames
    torch.cuda.synchronize()
    target=out/'measurements.jsonl'
    existing={}
    if target.exists():
        for line in target.read_text().splitlines():
            r=json.loads(line); existing[r['id'],r['mode']]=r
    with target.open('a',buffering=1) as dst, torch.inference_mode():
        for ordinal,row in enumerate(sample):
            modes=MODES[ordinal%4:]+MODES[:ordinal%4]
            for mode in modes:
                if (row['id'],mode) in existing: continue
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                path=root/'videos'/(row['video']+'.mp4')
                cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
                with av.open(str(path)) as container: n=sum(1 for _ in container.decode(video=0))
                idx=np.arange(n) if n<=5400 else np.linspace(0,n-1,5400,dtype=int)
                wanted=set(idx.tolist()); actual=[]; ds=[]; cs=[]; batch=[]
                def encode():
                    if mode!='query6':
                        x=torch.stack([transform(im) for im in batch]).cuda()
                        ds.append(dino(pixel_values=x).pooler_output.float().cpu().numpy())
                    if mode!='ktv6': cs.append(clip_images(batch))
                    batch.clear()
                with av.open(str(path)) as container:
                    for i,frame in enumerate(container.decode(video=0)):
                        if i in wanted:
                            batch.append(Image.fromarray(frame.to_ndarray(format='rgb24'))); actual.append(i)
                            if len(batch)==16: encode()
                        if i>=idx[-1]: break
                if batch: encode()
                assert actual==idx.tolist()
                base4=[]; base6=[]
                if ds:
                    d=np.concatenate(ds)
                    if mode=='ktv6': base6=ktv_frames(d,6)
                    else: base4=ktv_frames(d,4)
                if mode=='ktv6':
                    frames=load_frames(path,idx[base6].tolist())
                    c=clip_images(frames)
                else: c=np.concatenate(cs)
                words=row['question'].split(' ')
                query=' '.join(words[10:50]) if len(words)>40 else row['question']
                tokens=proc(text=[query],return_tensors='pt',padding=True,truncation=True)
                embedding=clip.get_text_features(**{k:v.cuda() for k,v in tokens.items()})
                embedding=embedding/embedding.norm(dim=-1,keepdim=True)
                scores=(torch.tensor(c,device='cuda',dtype=torch.float16)@embedding[0]).float().cpu().numpy()
                if mode=='ktv6':
                    ranked=np.argsort(-scores,kind='stable').tolist()
                    ranks=[ranked.index(i) for i in range(6)]; raw=idx[base6].tolist()
                else:
                    choice=select(mode,base4,base6,scores,idx/fps,row['question'])
                    raw=idx[choice['indices']].tolist(); ranks=choice['ranks']; frames=load_frames(path,raw)
                torch.cuda.synchronize(); selection_done=time.perf_counter()
                prediction=answer(row,frames,ranks); parsed=parse_answer(prediction)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                ref=references[row['id'],mode]
                record={'id':row['id'],'video':row['video'],'mode':mode,'candidates':len(idx),
                    'end_to_end_s':elapsed,'selection_pipeline_s':selection_done-start,
                    'answer_pipeline_s':time.perf_counter()-selection_done,'prediction':prediction,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'frame_indices':raw,'ranks':ranks,'visual_tokens':model.study_visual_tokens,
                    'frames_match_original':raw==ref['frame_indices'],
                    'ranks_match_original':ranks==ref['selection']['ranks'],
                    'answer_matches_original':parsed==ref['predicted_answer'],
                    'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
                assert len(set(raw))==6 and record['visual_tokens']==936
                dst.write(json.dumps(record)+'\n'); existing[row['id'],mode]=record
                print(json.dumps({'completed':len(existing),'total':32,**record}),flush=True)
    report={'n_videos':8,'measurements':len(existing),'modes':{},'parity':{}}
    for mode in MODES:
        rr=[r for r in existing.values() if r['mode']==mode]
        values=[r['end_to_end_s'] for r in rr]
        report['modes'][mode]={'mean_s':float(np.mean(values)),'median_s':float(np.median(values)),
            'min_s':min(values),'max_s':max(values), 'peak_allocated_mib':max(r['peak_allocated_mib'] for r in rr)}
        report['parity'][mode]={key:sum(r[key] for r in rr) for key in ('frames_match_original','ranks_match_original','answer_matches_original')}
    (out/'summary.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
