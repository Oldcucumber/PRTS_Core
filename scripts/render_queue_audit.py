"""Show successes and misses using frozen OCR and current field/task decisions."""
import argparse
import html
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw,ImageFont,ImageOps
from prts_core.queue_evidence import classify_rows
from prts_core.tasks import Tasks

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);rows=[];credits={}
    for folder in ('data','data-v2'):
        for item in json.loads((ROOT/f'outputs/stage2/queue-holdout/{folder}/sources.json').read_text(encoding='utf-8')):
            credits[item['id']]=item['source_metadata']
    for name in ('layout-dev-v4','layout-dev-v3'):
        for row in json.loads((ROOT/f'outputs/stage2/queue-holdout/{name}/results.json').read_text(encoding='utf-8'))['results']:
            c=row['case'];ocr=classify_rows(row['ocr']);observations=[]
            for target in c['positive_targets']+c['negative_targets']:
                state=Tasks();state.command(dict(action='wait',kind='number',target=target),0)
                observations.append(dict(target=target,event=state.observe(ocr,[],1),positive=target in c['positive_targets']))
            positive=sum((e['event'] or {}).get('type')=='target_observed' for e in observations if e['positive'])
            misses=[e['target'] for e in observations if e['positive'] and (e['event'] or {}).get('type')!='target_observed']
            false=[e['target'] for e in observations if not e['positive'] and (e['event'] or {}).get('type')=='target_observed']
            candidates=[e['target'] for e in observations if not e['positive'] and (e['event'] or {}).get('type')=='target_candidate']
            rows.append(dict(case=c,ocr=ocr,confirmed_positive=positive,misses=misses,false_positives=false,false_candidates=candidates))
    canvas=Image.new('RGB',(1800,2120),'#eef2f7');draw=ImageDraw.Draw(canvas)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    draw.text((25,20),'PRTS Core · 真实叫号照片：成功与未通过案例',font=font(33),fill='#16283f')
    draw.text((25,72),'当前字段决策重放既有 OCR；这些素材已参与调整，不能再计为独立测试。',font=font(23),fill='#53627b')
    colors={'called':'#20bf50','counter':'#ed7624','waiting':'#edc321','unconfirmed':'#b73ab8'}
    for i,row in enumerate(rows):
        x=20+(i%2)*890;y=125+(i//2)*480;c=row['case'];info=credits[c['id']];ext=info['extmetadata']
        draw.rounded_rectangle((x,y,x+870,y+464),12,fill='white',outline='#c6d0de')
        draw.text((x+15,y+10),c['id'],font=font(24),fill='#16283f')
        image=Image.open(ROOT/c['source']).convert('RGB');d=ImageDraw.Draw(image)
        for e in row['ocr']:
            if e['queue_role'] in colors:
                pts=[tuple(p) for p in e['box']];d.line(pts+[pts[0]],fill=colors[e['queue_role']],width=max(3,image.width//260))
        preview=ImageOps.contain(image,(835,298));canvas.paste(preview,(x+(870-preview.width)//2,y+48+(298-preview.height)//2))
        count=len(c['positive_targets'])
        text=f"确认 {row['confirmed_positive']} / {count} 个正目标" if count else '无可核验正目标；只检查指定反例'
        text+=f"；误确认 {len(row['false_positives'])} / {len(c['negative_targets'])}"
        draw.text((x+15,y+354),text,font=font(23),fill='#aa422e' if row['misses'] or row['false_positives'] else '#197159')
        detail='仍未确认：'+', '.join(row['misses']) if row['misses'] else '窗口、前后缀或非目标数字未触发到号提醒'
        draw.text((x+15,y+390),detail,font=font(21),fill='#53627b')
        author=html.unescape(re.sub('<[^>]+>','',ext['Artist']['value']))
        draw.text((x+15,y+432),author+' / Wikimedia Commons / '+ext['LicenseShortName']['value'],font=font(16),fill='#53627b')
    draw.text((25,2053),'绿框：已叫字段；橙框：柜台；紫框：栏目未确认。不是到号变化视频、真实广播或 Apple 测试。',font=font(22),fill='#53627b')
    canvas.save(a.output/'review-sheet.png')
    (a.output/'results.json').write_text(json.dumps(dict(scope=__doc__,results=rows,
        positive=sum(r['confirmed_positive'] for r in rows),misses=sum(len(r['misses']) for r in rows),
        false_positives=sum(len(r['false_positives']) for r in rows),
        false_candidates=sum(len(r['false_candidates']) for r in rows)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(photos=len(rows),confirmed=sum(r['confirmed_positive'] for r in rows),misses=sum(len(r['misses']) for r in rows),
                         false_positives=sum(len(r['false_positives']) for r in rows),false_candidates=sum(len(r['false_candidates']) for r in rows))))

if __name__=='__main__':main()
