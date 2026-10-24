"""Official DINO sampling/preprocessing and CLIP features on the same candidates."""
import argparse
import json
from pathlib import Path
import time
import av
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import Dinov2Model, CLIPModel, CLIPProcessor
from selection import ktv_frames


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True); p.add_argument('--split',default='pilot')
    p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=1)
    a=p.parse_args(); root=Path(a.root)
    rows=json.loads((root/(a.split+'.json')).read_text())
    vids=sorted({r['video'] for r in rows})[a.shard::a.shards]
    dest=root/'features'; dest.mkdir(exist_ok=True)
    pins=json.loads((root/'asset_pins.json').read_text())
    torch.set_num_threads(4)
    dino=Dinov2Model.from_pretrained(pins['facebook/dinov2-large']['path']).eval().cuda()
    clip=CLIPModel.from_pretrained(pins['openai/clip-vit-large-patch14']['path']).eval().cuda().half()
    proc=CLIPProcessor.from_pretrained(pins['openai/clip-vit-large-patch14']['path'])
    transform=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    for number,vid in enumerate(vids):
        target=dest/(vid+'.npz')
        if target.exists(): continue
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        cap=cv2.VideoCapture(str(root/'videos'/(vid+'.mp4')))
        metadata_n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
        # Container frame counts can include frames the decoder never emits.
        # Count with the same decoder used for features and inference before
        # constructing the shared pool, including the >5400 sampling case.
        count_start=time.perf_counter()
        with av.open(str(root/'videos'/(vid+'.mp4'))) as container:
            n=sum(1 for _ in container.decode(video=0))
        frame_count_validation_s=time.perf_counter()-count_start
        if n!=metadata_n:
            print(json.dumps({'frame_count_correction':vid,'metadata_frames':metadata_n,
                              'decoded_frames':n}),flush=True)
        if n<6 or fps<=0: raise ValueError(f'Invalid video {vid}, {n}, {fps}')
        idx=np.arange(n) if n<=5400 else np.linspace(0,n-1,5400,dtype=int)
        wanted=set(idx.tolist()); ds=[]; cs=[]; frames=[]; actual=[]; decode_s=0.; dino_s=0.; clip_s=0.
        def batch():
            nonlocal dino_s,clip_s
            torch.cuda.synchronize(); t=time.perf_counter()
            with torch.inference_mode():
                x=torch.stack([transform(im) for im in frames]).cuda()
                ds.append(dino(pixel_values=x).pooler_output.float().cpu().numpy())
            torch.cuda.synchronize(); dino_s+=time.perf_counter()-t; t=time.perf_counter()
            with torch.inference_mode():
                x=proc(images=frames,return_tensors='pt')['pixel_values'].cuda().half()
                feat=clip.get_image_features(pixel_values=x)
                feat=feat/feat.norm(dim=-1,keepdim=True)
                cs.append(feat.float().cpu().numpy())
            torch.cuda.synchronize(); clip_s+=time.perf_counter()-t; frames.clear()
        t=time.perf_counter()
        with av.open(str(root/'videos'/(vid+'.mp4'))) as container:
            for i,frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    frames.append(Image.fromarray(frame.to_ndarray(format='rgb24'))); actual.append(i)
                    if len(frames)==16:
                        decode_s+=time.perf_counter()-t; batch(); t=time.perf_counter()
                if i>=idx[-1]: break
        decode_s+=time.perf_counter()-t
        if frames: batch()
        if actual != idx.tolist(): raise ValueError(f'Decoded candidate mismatch: {vid}')
        d,c=np.concatenate(ds),np.concatenate(cs)
        t=time.perf_counter(); b4=ktv_frames(d,4); k4_s=time.perf_counter()-t
        t=time.perf_counter(); b6=ktv_frames(d,6); k6_s=time.perf_counter()-t
        # Native KTV CLIP cost is six images, not the dense candidate pool.
        baseline_images=[]; raw_base=set(idx[b6].tolist())
        t=time.perf_counter()
        with av.open(str(root/'videos'/(vid+'.mp4'))) as container:
            for j,frame in enumerate(container.decode(video=0)):
                if j in raw_base: baseline_images.append(Image.fromarray(frame.to_ndarray(format='rgb24')))
                if j>=max(raw_base): break
        baseline_clip_decode_s=time.perf_counter()-t
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.inference_mode():
            x=proc(images=baseline_images,return_tensors='pt')['pixel_values'].cuda().half()
            baseline_clip=clip.get_image_features(pixel_values=x)
            baseline_clip=baseline_clip/baseline_clip.norm(dim=-1,keepdim=True)
        torch.cuda.synchronize(); baseline_clip_s=time.perf_counter()-t
        # Batch-size changes can introduce small floating point differences.
        baseline_clip_max_abs=float(np.max(np.abs(baseline_clip.float().cpu().numpy()-c[b6])))
        meta={'video':vid,'fps':fps,'total_frames':n,'candidates':len(idx),
              'metadata_frames':metadata_n,'frame_count_validation_s':frame_count_validation_s,
              'decode_s':decode_s,'dino_s':dino_s,'clip_image_s':clip_s,
              'kmeans4_s':k4_s,'kmeans6_s':k6_s,'feature_wall_s':time.perf_counter()-start,
              'baseline_clip_decode_s':baseline_clip_decode_s,'baseline_clip_s':baseline_clip_s,
              'baseline_clip_batch_max_abs':baseline_clip_max_abs,
              'feature_peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
              'feature_peak_reserved_mb':torch.cuda.max_memory_reserved()/1024**2}
        temp=target.with_suffix('.partial.npz')
        np.savez(temp,dino=d,clip=c,indices=idx,times=idx/fps,base4=b4,base6=b6,metadata=json.dumps(meta))
        temp.replace(target)
        print(json.dumps({'progress':f'{number+1}/{len(vids)}',**meta}),flush=True)

if __name__=='__main__': main()
