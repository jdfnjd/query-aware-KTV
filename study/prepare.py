"""Download once, inventory available validation videos, freeze pilot membership."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import random
import subprocess
import re
import time
import zipfile


def resumable_http_get(url, temp_file, *, expected_size=None, **kwargs):
    """Use curl transport while preserving Hugging Face's cache and file locks.

    Requests transfers repeatedly stalled on this host. curl successfully
    transferred the same official wheels; preserve partial cache blobs on retry.
    All repos in this study are public and signed blob URLs come from HF metadata.
    """
    temp_file.flush()
    target=Path(temp_file.name)
    for attempt in range(1,31):
        size=target.stat().st_size
        if expected_size is not None and size==expected_size: return
        print(f'Download bytes {size}/{expected_size}, attempt {attempt}',flush=True)
        result=subprocess.run(['curl','-sS','-fL','--connect-timeout','20','--max-time','120',
            '--speed-time','25','--speed-limit','1024','-C','-','-o',str(target),url])
        size=target.stat().st_size
        if result.returncode==0 and (expected_size is None or size==expected_size):
            temp_file.seek(0,2); return
        time.sleep(3)
    raise RuntimeError(f'Download incomplete after retry limit: {size}/{expected_size}')


def check_quota(required_bytes, reserve_bytes=10*1024**3):
    """Read NFS user quota in KiB, not the filesystem's shared free capacity."""
    declared=os.environ.get('STUDY_STORAGE_BUDGET_GIB')
    filesystem=os.environ.get('STUDY_QUOTA_FILESYSTEM')
    if filesystem:
        quota=subprocess.check_output(['quota','-w','-v'],text=True)
        match=re.search(re.escape(filesystem)+r'\s+(\d+)\*?\s+(\d+)\s+(\d+)',quota)
        if not match: raise RuntimeError('Cannot verify the configured user quota')
        used,soft,hard=map(int,match.groups()); allowance=(min(soft,hard)-used)*1024
    elif declared:
        base=Path(os.environ.get('STUDY_BASE',str(Path(__file__).resolve().parents[1]/'.runtime')))
        # Budget is an explicit allowance supplied after checking user quota.
        # Subtract existing runtime files on every check, including partial downloads.
        used=sum(p.stat().st_size for p in base.rglob('*') if p.is_file() and not p.is_symlink())
        allowance=min(int(float(declared)*1024**3)-used,shutil.disk_usage(base).free)
    else:
        raise RuntimeError('Set STUDY_QUOTA_FILESYSTEM or STUDY_STORAGE_BUDGET_GIB after checking quota')
    if required_bytes+reserve_bytes>allowance:
        raise RuntimeError(f'Insufficient quota: {allowance} bytes available, {required_bytes} planned')
    print('Quota headroom bytes',allowance,'planned additional bytes',required_bytes,flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--mapping', required=True)
    a = p.parse_args()
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    import huggingface_hub.file_download as hf_download
    hf_download.http_get=resumable_http_get
    api = HfApi()
    repos = ['liuhaotian/llava-v1.6-vicuna-7b', 'facebook/dinov2-large',
             'openai/clip-vit-large-patch14', 'openai/clip-vit-large-patch14-336']
    pin_file=root/'asset_pins.json'
    pins = json.loads(pin_file.read_text()) if pin_file.exists() else {}
    for repo in repos:
        info = api.model_info(repo, revision=pins.get(repo,{}).get('revision'), files_metadata=True)
        names=[s.rfilename for s in info.siblings]
        has_safe=any(n.endswith('.safetensors') for n in names)
        weights=['*.safetensors'] if has_safe else ['pytorch_model*.bin']
        pins[repo] = {'revision': info.sha,'weight_format':'safetensors' if has_safe else 'pytorch_bin'}
        pin_file.write_text(json.dumps(pins,indent=2))
        weight_bytes=sum(s.size or 0 for s in info.siblings if (
            s.rfilename.endswith('.safetensors') if has_safe else
            s.rfilename.startswith('pytorch_model') and s.rfilename.endswith('.bin')))
        check_quota(weight_bytes)
        print('Downloading model', repo, info.sha, flush=True)
        pins[repo]['path'] = snapshot_download(repo, revision=info.sha,
            allow_patterns=['*.json', '*.model', '*.txt']+weights,
            ignore_patterns=['tf_model*','flax_model*'], max_workers=2)
        pin_file.write_text(json.dumps(pins,indent=2))
    info = api.dataset_info('lmms-eval/NExTQA',revision=pins.get('dataset',{}).get('revision'))
    pins['dataset'] = {'repo':'lmms-eval/NExTQA', 'revision':info.sha}
    (root/'asset_pins.json').write_text(json.dumps(pins, indent=2))
    check_quota(7*1024**3)
    archive = hf_hub_download('lmms-eval/NExTQA','videos.zip',repo_type='dataset',revision=info.sha)
    rows = list(csv.DictReader(open(a.annotations)))
    mapping = json.load(open(a.mapping))
    videos = root/'videos'; videos.mkdir(exist_ok=True)
    wanted = {r['video'] for r in rows}
    reverse = {str(v).split('/')[-1]:str(k) for k,v in mapping.items() if str(k) in wanted}
    with zipfile.ZipFile(archive) as z:
        inventory = [{'name':i.filename,'size':i.file_size} for i in z.infolist()]
        (root/'archive_inventory.json').write_text(json.dumps(inventory))
        total = sum(i['size'] for i in inventory)
        if total > 40*1024**3:
            raise RuntimeError('Archive exceeds planned 40 GiB extraction ceiling')
        check_quota(total)
        for i in z.infolist():
            if Path(i.filename).suffix.lower() != '.mp4': continue
            stem = Path(i.filename).stem
            vid = stem if stem in wanted else reverse.get(stem)
            if vid is None: continue
            target = videos/(vid+'.mp4')
            if target.exists() and target.stat().st_size == i.file_size: continue
            # Flatten into known numeric IDs, never trust archive paths.
            if not vid.isdigit(): raise ValueError('Unexpected video ID')
            with z.open(i) as src, open(target.with_suffix('.partial'),'wb') as dst:
                import shutil
                shutil.copyfileobj(src,dst)
            target.with_suffix('.partial').replace(target)
    available, missing = [], []
    for r in rows:
        r = dict(r)
        r.update(id=r['video']+'_'+r['qid'], answer=int(r['answer']),
                 candidates=[r['a'+str(i)] for i in range(5)])
        (available if (videos/(r['video']+'.mp4')).exists() else missing).append(r)
    if len(available) < 100:
        raise RuntimeError(f'Only {len(available)} validation questions have videos; inspect inventory')
    # Proportional stratification by official subtype, frozen before inference.
    groups = {}
    for r in available: groups.setdefault(r['type'],[]).append(r)
    counts = {t:100*len(g)/len(available) for t,g in groups.items()}
    allocations = {t:int(n) for t,n in counts.items()}
    for t in sorted(counts,key=lambda t:(-(counts[t]-allocations[t]),t))[:100-sum(allocations.values())]:
        allocations[t] += 1
    rng = random.Random(20260927); pilot=[]
    for t in sorted(groups):
        g = sorted(groups[t],key=lambda r:r['id']); rng.shuffle(g)
        pilot.extend(g[:allocations[t]])
    pilot = sorted(pilot,key=lambda r:r['id'])
    for name,data in [('pilot',pilot),('smoke',pilot[:1]),('full',available),('missing',missing)]:
        (root/(name+'.json')).write_text(json.dumps(data,indent=2))
    summary = {'official_validation_questions':len(rows),'available_questions':len(available),
               'missing_questions':len(missing),'available_videos':len({r['video'] for r in available}),
               'pilot_types':allocations,'seed':20260927,
               'annotations_sha256':hashlib.sha256(Path(a.annotations).read_bytes()).hexdigest()}
    (root/'data_summary.json').write_text(json.dumps(summary,indent=2)); print(summary,flush=True)

if __name__ == '__main__': main()
