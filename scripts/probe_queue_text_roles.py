"""Classify observed OCR row roles without generating or expecting queue numbers."""
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prts_core.native_vlm import NativeVLM
PROMPT=('识别叫号屏文字的用途。下列每条 OCR 有固定行号 id 和归一化中心坐标 x,y。'
        '选出当前已经叫到或可取餐的号码所在行，排除等候号码、柜台编号、价格、广告和汇率。'
        '只返回原有行号，不识别或改写号码。返回单个 JSON 对象：'
        '{"called_rows":[行号],"waiting_rows":[行号],"counter_rows":[行号]}。'
        '不明确时该数组为空。OCR：\n')


def main():
    out=ROOT/'outputs/stage2/queue-holdout/text-role-probe-v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    log=(out/'native-stderr.log').open('w');os.dup2(log.fileno(),2)
    source=json.loads((ROOT/'outputs/stage2/queue-holdout/first-pass-v1/results.json').read_text(encoding='utf-8'))
    import cv2
    model=NativeVLM(ROOT/'models',ROOT/'outputs/prts-native-build/libprts_vlm.dll',image_slices=1,
        language_file='MiniCPM-V-4_6-Q5_K_M.gguf',projector_file='mmproj-matrix-q8_0.gguf')
    rows=[]
    for case in source['results']:
        frame=cv2.imread(str(ROOT/case['case']['source']));h,w=frame.shape[:2]
        observations=[dict(id=i,text=e['text'],x=round(sum(p[0] for p in e['box'])/4/w,3),y=round(sum(p[1] for p in e['box'])/4/h,3)) for i,e in enumerate(case['ocr']) if e['score']>=.7]
        prompt=PROMPT+json.dumps(observations,ensure_ascii=False)
        r=model.generate(prompt,max_tokens=160,raw_prompt=True)
        match=re.search(r'\{.*\}',r['text'],re.S)
        try:parsed=json.loads(match[0]) if match else None
        except json.JSONDecodeError:parsed=None
        row=dict(case=case['case']['id'],prompt=prompt,response=r,parsed=parsed);rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(status='development role classification candidate; not production',results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    model.close()


if __name__=='__main__':main()
