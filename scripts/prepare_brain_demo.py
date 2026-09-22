"""Build a declared replay from real attributed road frames and queue/bus photos.

This is a scenario montage, not a recording of one person's continuous journey.
No target digits are painted onto the camera input. Commands are scripted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    road=ROOT/'outputs/stage2/perception/data/dev-turn'
    road_meta=json.loads((road/'manifest.json').read_text(encoding='utf-8'))
    bank='outputs/stage2/queue-holdout/data/commons-bank-queue.jpg'
    segments=[dict(start=0,end=24,kind='road',label='胸前相机道路片段，循环两次'),
              dict(start=24,end=84,file='outputs/stage2/bus-generalization/data/commons-manchester-101.jpg',label='真实101路照片，等待101'),
              dict(start=84,end=144,file=bank,label='真实叫号屏照片，等待3（柜台反例）'),
              dict(start=144,end=204,file=bank,label='同一画面，改为等待C002'),
              dict(start=204,end=244,file=bank,label='临时询问屏幕内容'),
              dict(start=244,end=294,file='outputs/stage2/queue-holdout/data-v2/masmitra.jpg',label='等待B003，取消后重新等待'),
              dict(start=294,end=330,kind='road',label='返回道路片段，循环三次')]
    writer=cv2.VideoWriter(str(out/'montage-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(1280,960))
    photos={s['file']:cv2.imread(str(ROOT/s['file'])) for s in segments if 'file' in s}
    for i in range(660):
        t=i/2;s=next(s for s in segments if s['start']<=t<s['end'])
        if s.get('kind')=='road':
            idx=round(((t-s['start'])%12)*15)%len(road_meta['frames'])
            bgr=cv2.imread(str(road/road_meta['frames'][idx]['file']))
        else:bgr=photos[s['file']]
        if bgr is None:raise FileNotFoundError(s)
        h,w=bgr.shape[:2];scale=min(1280/w,960/h);nw,nh=round(w*scale),round(h*scale)
        canvas=np.zeros((960,1280,3),np.uint8);x,y=(1280-nw)//2,(960-nh)//2
        canvas[y:y+nh,x:x+nw]=cv2.resize(bgr,(nw,nh));writer.write(canvas)
    writer.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(out/'montage-raw.mp4'),'-c:v','libx264','-crf','17',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'montage.mp4')],check=True)
    scenario=dict(source=str((out/'montage.mp4').relative_to(ROOT)) if out.is_absolute() else str(out/'montage.mp4'),
        start=0,end=330,navigate=True,audit_kind='brain_montage',image_segments=segments,
        user_audio=[dict(at=87,file='outputs/stage2/streaming/queue-montage-input-v1/wait3.wav',role='user',
                         synthetic=True,text='帮我等三号，叫到时提醒我。')],
        text_commands=[dict(at=26,text='等101路公交车，到了提醒我'),dict(at=145,text='我的号码是C002，叫到时提醒我'),
                       dict(at=207,text='这块屏幕上写了什么？'),dict(at=246,text='等B003号，叫到时提醒我'),
                       dict(at=252,text='取消等待'),dict(at=257,text='等B003号，叫到时提醒我')],
        provenance=__doc__,expected_checks=['101提醒一次','柜台3不提醒','换目标C002提醒','临时问答保留等待任务',
            '取消后的旧B003结果不提醒','新B003任务独立判断','导航持续产生实际模型结果'])
    (out/'scenario.json').write_text(json.dumps(scenario,ensure_ascii=False,indent=2),encoding='utf-8')
    credits=dict(provenance=__doc__,sources=[],road=road_meta,segments=segments)
    for p in [ROOT/'outputs/stage2/queue-holdout/data/sources.json',ROOT/'outputs/stage2/queue-holdout/data-v2/sources.json',
              *sorted((ROOT/'outputs/stage2/bus-generalization/data').glob('*sources*.json'))]:
        credits['sources'].append(dict(manifest=str(p.relative_to(ROOT)),metadata=json.loads(p.read_text(encoding='utf-8'))))
    credits['input_sha256']=hashlib.sha256((out/'montage.mp4').read_bytes()).hexdigest()
    (out/'sources.json').write_text(json.dumps(credits,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),duration_s=330,frames=660)))


if __name__=='__main__':main()
