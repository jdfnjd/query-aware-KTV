"""Additional source-video context for the six preselected hybrid cases."""
import json
import os
from pathlib import Path
import av
from PIL import Image, ImageDraw, ImageFont

root=Path(os.environ.get('STUDY_BASE', str(Path(__file__).resolve().parents[1]/'.runtime')))/'data'
out=root/'cases'/'context_review'; out.mkdir(exist_ok=True)
windows={'3131380536_7':[(28,34)],'4488066715_2':[(0,5)],
         '2871995580_9':[(54,59),(71,77)],'11584566583_5':[(5,9)],
         '6297019062_6':[(9,12),(32,35)]}
font=ImageFont.truetype('DejaVuSans.ttf',16)
for qid,ranges in windows.items():
    row=json.loads((root/'cases'/'subset400'/(qid+'.json')).read_text())[0]
    fps=row['feature_metrics']['fps']
    indices=sorted({round(t*fps) for lo,hi in ranges for t in [lo+j*.5 for j in range(int((hi-lo)*2)+1)]})
    frames={}
    with av.open(str(root/'videos'/(row['video']+'.mp4'))) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in indices: frames[i]=frame.to_image()
            if i>=max(indices): break
    canvas=Image.new('RGB',(1200,40+((len(frames)+3)//4)*205),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((8,8),qid+' | source context at 0.5-second intervals',font=font,fill='black')
    for j,(idx,im) in enumerate(frames.items()):
        x=(j%4)*300; y=40+(j//4)*205
        im.thumbnail((290,175)); canvas.paste(im,(x,y))
        draw.text((x+5,y+178),f'{idx/fps:.2f}s',font=font,fill='black')
    canvas.save(out/(qid+'.jpg'),quality=93)
    print(qid,len(frames))
