"""Freeze retained bus/changed-display checks before the selected VLM sees them."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from PIL import Image,ImageDraw


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((ROOT/'outputs/stage2/bus-generalization/vision-q8-regression-v1/results.json').read_text(encoding='utf-8'))
    frames=[];cases=[];credits=[]
    for row in baseline['results']:
        c=row['case'];source=Path(row['source'])
        # Fixed retained positive times from the old pipeline, not VLM-selected successes.
        times=[e['observation_s'] for e in row['events'] if e['requested_target']==c['expected_route']][:1]
        if c['id']=='commons-citybus-20':times=[0,5,10]
        for idx,t in enumerate(times):
            if source.suffix.lower() in ('.jpg','.png','.jpeg'):bgr=cv2.imread(str(source))
            else:
                cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();cap.release()
                if not ok:raise RuntimeError('Cannot decode frozen frame')
            if c.get('roi'):
                h,w=bgr.shape[:2];x1,y1,x2,y2=c['roi'];bgr=bgr[int(y1*h):int(y2*h),int(x1*w):int(x2*w)].copy()
            for x1,y1,x2,y2 in c.get('ignore_regions',[]):
                h,w=bgr.shape[:2];bgr[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]=0
            name=f"{c['id']}-{idx}.png";file=out/name;cv2.imwrite(str(file),bgr)
            path=file.relative_to(ROOT).as_posix() if file.is_absolute() else file.as_posix()
            tag=f"{c['id']} @ {t}s";frames.append((tag,bgr))
            targets=[(c['expected_route'],True),(c['negative_targets'][0],False)]
            if c['id']=='commons-utrecht-8':targets[-1]=('4833',False)
            if c['id']=='commons-citybus-20':targets[-1]=('20A',False)
            for target,expected in targets:
                cases.append(dict(id=f"{c['id']}-{idx}-{target}",source=path,kind='bus',target=target,expected=expected))
            credits.append(dict(file=path,original=str(source),source_s=t,roi=c.get('roi'),ignore_regions=c.get('ignore_regions'),
                                sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    manifest=dict(scope='Retained development sources; positive time selected by the previous OCR pipeline, not blind data. Three actual video times test changed display. Frames and goals frozen before this VLM pass.',cases=cases,inputs=credits)
    (out/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1800,450*((len(frames)+2)//3)),(20,25,34));draw=ImageDraw.Draw(sheet)
    for i,(tag,bgr) in enumerate(frames):
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((590,420))
        x,y=(i%3)*600,(i//3)*450;sheet.paste(im,(x+(600-im.width)//2,y+25));draw.text((x+8,y+7),tag,fill='white')
    sheet.save(out/'contact-sheet.jpg')
    print(json.dumps(dict(cases=len(cases),output=str(out))))


if __name__=='__main__':main()
