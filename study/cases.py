"""Create auditable frame contact sheets for paired prediction flips."""
import argparse
import json
from pathlib import Path
import textwrap
import av
from PIL import Image, ImageDraw, ImageFont
from selection import MODES


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--split',default='pilot')
    a=p.parse_args(); root=Path(a.root)
    summary=json.loads((root/(a.split+'_summary.json')).read_text())
    rows={}
    for path in (root/'results').glob(a.split+'_*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line); rows[r['id'],r['mode']]=r
    dest=root/'cases'/a.split; dest.mkdir(parents=True,exist_ok=True)
    chosen=set()
    # Deterministically select one case per direction, subtype group, and control.
    for mode,flips in summary['cases'].items():
        for direction,ids in flips.items():
            for group in ('C','T','D'):
                matching=[i for i in ids if rows[i,mode]['type'].startswith(group)]
                if matching: chosen.add(sorted(matching)[0])
    manifest=[]
    try: font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError: font=ImageFont.load_default()
    for qid in sorted(chosen):
        rr=[rows[qid,m] for m in MODES]; base=rr[0]
        wanted=set(i for r in rr for i in r['frame_indices']); frames={}
        with av.open(str(root/'videos'/(base['video']+'.mp4'))) as container:
            for i,frame in enumerate(container.decode(video=0)):
                if i in wanted: frames[i]=frame.to_image()
                if i>=max(wanted): break
        heading=f"{qid} | type {base['type']} | correct answer {'ABCDE'[base['answer']]}\n{base['question']}\n"
        heading+=' | '.join(f"{'ABCDE'[i]}: {c}" for i,c in enumerate(base['candidates']))
        lines=[line for paragraph in heading.splitlines() for line in textwrap.wrap(paragraph,width=145)]
        header_height=max(170,len(lines)*23+30)
        canvas=Image.new('RGB',(1440,header_height+4*217+15),'white'); draw=ImageDraw.Draw(canvas)
        y=10
        for line in lines:
            draw.text((10,y),line,fill='black',font=font); y+=23
        y=header_height
        for r in rr:
            draw.text((10,y),f"{r['mode']} | prediction: {r['prediction'][:90]} | correct={r['correct']}",fill='black',font=font)
            for j,(idx,t) in enumerate(zip(r['frame_indices'],r['frame_times'])):
                im=frames[idx].copy(); im.thumbnail((230,150))
                canvas.paste(im,(j*240+5,y+28))
                draw.text((j*240+5,y+183),f"frame {idx}, {t:.2f}s",fill='black',font=font)
            y+=217
        canvas.save(dest/(qid+'.jpg'),quality=93)
        (dest/(qid+'.json')).write_text(json.dumps(rr,indent=2))
        manifest.append({'id':qid,'image':qid+'.jpg','type':base['type'],
                         'question':base['question'],'review_status':'pending_visual_review'})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'{len(manifest)} case sheets saved to {dest}')

if __name__=='__main__': main()
