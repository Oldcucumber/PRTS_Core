"""Render actual bus-test evidence into a review sheet and source-time videos."""
import argparse
import json
from pathlib import Path
import subprocess
import html
import re
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
FONT_PATH='C:/Windows/Fonts/msyh.ttc'
SOURCES={}
for name in ['sources.json','holdout-v2-sources.json','holdout-v3-sources.json']:
    p=ROOT/'outputs/stage2/bus-generalization/data'/name
    if p.exists():
        for source in json.loads(p.read_text(encoding='utf-8')):
            meta=source['source_metadata'];ext=meta['extmetadata']
            SOURCES[source['id']]=dict(author=html.unescape(re.sub('<[^>]+>','',ext['Artist']['value'])),
                license=ext['LicenseShortName']['value'],license_url=ext['LicenseUrl']['value'],
                source_url=meta['descriptionurl'],modification='Resize and annotations from actual PRTS evaluation; source imagery unchanged.')


def font(size):return ImageFont.truetype(FONT_PATH,size)


def frame_at(source,t):
    if source.suffix.lower() in ('.jpg','.png'):return cv2.imread(str(source))
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();cap.release()
    if not ok:raise RuntimeError(f'Frame unavailable: {source} {t}')
    return bgr


def card(result,t,sample,width=1200,height=850):
    bgr=frame_at(ROOT/result['source'],t)
    case=result['case']
    if case.get('roi'):
        h,w=bgr.shape[:2];x1,y1,x2,y2=case['roi'];bgr=bgr[int(y1*h):int(y2*h),int(x1*w):int(x2*w)].copy()
    for x1,y1,x2,y2 in case.get('ignore_regions',[]):
        h,w=bgr.shape[:2];bgr[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]=0
    h,w=bgr.shape[:2]
    im=Image.new('RGB',(width,height),'#f5f7fa');d=ImageDraw.Draw(im)
    label=f"线路 {case['expected_route']}" if case['expected_route'] is not None else '反例：无可确认的目标线路'
    title=label+' · '+('静态照片' if case.get('media_type')=='image' else f'源视频 {t:.1f} 秒')
    d.text((26,18),title,font=font(32),fill='#172d45')
    photo=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));photo.thumbnail((width-52,540))
    px,py=26,75;im.paste(photo,(px,py));sx,sy=photo.width/w,photo.height/h
    for e in sample.get('ocr',[]):
        if e['score']<.45 and not e.get('route_candidate'):continue
        color='#14866d' if e.get('route_verified') else '#bf8627' if e.get('route_candidate') else '#77879a'
        pts=[(int(px+x*sx),int(py+y*sy)) for x,y in e['box']]
        d.line(pts+[pts[0]],fill=color,width=3)
    events=[e for e in result['events'] if e['observation_s']<=t]
    positive=[e for e in events if e['requested_target']==result['case']['expected_route']]
    latest=positive[-1] if positive else None
    status=latest['text'] if latest else '尚未确认目标'
    color='#14866d' if latest and latest['type']=='target_observed' else '#a76614'
    d.text((26,635),status,font=font(27),fill=color)
    negatives=result['case']['negative_targets']
    bad=[x for x in result['false_positive_targets'] if any(e['requested_target']==x and e['type']=='target_observed' for e in events)]
    d.text((26,682),'干扰目标：'+' / '.join(negatives)+'；确认误报：'+('、'.join(bad) if bad else '无'),font=font(22),fill='#334a63')
    d.text((26,721),f"此帧实际处理 {sample.get('processing_ms',0)/1000:.2f} 秒 · 绿色：交叉验证文字；灰色：其他文字",font=font(19),fill='#53657d')
    d.text((26,758),'离线抽帧效果回放，不是现场响应速度。候选提示不算目标确认。',font=font(19),fill='#53657d')
    source=SOURCES.get(result['case']['id'],{})
    credit=f"作者：{source.get('author','见来源清单')} · {source.get('license','')} · 改动：缩放及标注；来源见 source-credits.json"
    d.text((26,793),credit,font=font(18),fill='#53657d')
    return im


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,action='append',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--video',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);cards=[]
    for path in args.results:
        report=json.loads(path.read_text(encoding='utf-8'))
        for result in report['results']:
            evidence=json.loads((path.parent/(result['case']['id']+'-evidence.json')).read_text(encoding='utf-8'))
            target=result['case']['expected_route'];match=next((e for e in result['events'] if e['requested_target']==target),None)
            t=match['observation_s'] if match else evidence[0]['source_s']
            sample=min(evidence,key=lambda x:abs(x['source_s']-t));im=card(result,t,sample)
            im.save(args.output/(result['case']['id']+'.png'));cards.append(im)
            if args.video and result['case'].get('media_type')!='image':
                raw=args.output/(result['case']['id']+'-raw.mp4');final=args.output/(result['case']['id']+'.mp4')
                writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),result['sample_fps'],(1200,850))
                for sample in evidence:
                    view=card(result,sample['source_s'],sample)
                    writer.write(cv2.cvtColor(np.asarray(view),cv2.COLOR_RGB2BGR))
                writer.release()
                subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True)
    sheet=Image.new('RGB',(1600,round(len(cards)/2+.49)*570),'#e4eaf0')
    for i,im in enumerate(cards):im.thumbnail((790,560));sheet.paste(im,((i%2)*800,(i//2)*570))
    sheet.save(args.output/'review-sheet.png')
    (args.output/'source-credits.json').write_text(json.dumps(dict(sources=SOURCES,
        composite_license='CC BY-SA 4.0',note='Share-alike applies to the annotated composite/media, not to unrelated program source.'),
        ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
