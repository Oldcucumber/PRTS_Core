"""Evaluate target-blind visual field classification; never use generated digits."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from prts_core.native_vlm import NativeVLM
from prts_core.tasks import identifiers

PROMPT=('黄色框中的文字是什么字段？只回答一个类别：叫到号码、等待号码、柜台号码、其他、混合、无法确定。'
        '只有正在服务或可以取餐的号码才是叫到号码。纸质排队票、取号机分配的新票号不是已经叫到。'
        '如果黄色框同时包括叫到号码和柜台号码，回答混合。不要抄写号码。')

def main():
    out=ROOT/'outputs/stage2/queue-holdout/field-role-probe-v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    (out/'prompt.txt').write_text(PROMPT,encoding='utf-8')
    (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    log=(out/'native-stderr.log').open('w');os.dup2(log.fileno(),2)
    report=json.loads((ROOT/'outputs/stage2/queue-holdout/layout-dev-v3/results.json').read_text(encoding='utf-8'))
    model=NativeVLM(ROOT/'models',ROOT/'outputs/prts-native-build/libprts_vlm.dll',image_slices=1,
                    language_file='MiniCPM-V-4_6-Q5_K_M.gguf',projector_file='mmproj-matrix-q8_0.gguf')
    rows=[]
    for case in report['results']:
        if case['case']['id']=='jtrust':continue
        file=ROOT/case['case']['source'];assert hashlib.sha256(file.read_bytes()).hexdigest()==case['case']['sha256']
        source=cv2.imread(str(file));height,width=source.shape[:2]
        for i,e in enumerate(case['ocr']):
            if not identifiers(e['text']) or e['score']<.7:continue
            if case['case']['id']=='masmitra' and e['text'] not in ('LOKET 2','A-002','B-003'):continue
            box=np.asarray(e['box'],np.float32);x1,y1=box.min(0);x2,y2=box.max(0);w=x2-x1;h=y2-y1
            left,top=max(0,int(x1-w)),max(0,int(y1-3*h));right,bottom=min(width,int(x2+w)),min(height,int(y2+2*h))
            crop=source[top:bottom,left:right].copy();local=np.round(box-[left,top]).astype(np.int32)
            cv2.polylines(crop,[local],True,(0,255,255),max(2,round(max(crop.shape[:2])/250)))
            preview=out/(case['case']['id']+'-'+str(i)+'.jpg');cv2.imwrite(str(preview),crop)
            response=model.generate(PROMPT,crop,max_tokens=32,raw_prompt=True)
            row=dict(case=case['case']['id'],field=e['text'],previous_role=e['queue_role'],image=preview.name,response=response)
            rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
            (out/'results.json').write_text(json.dumps(dict(scope='Diagnostic on already inspected development images. No target in prompt; no generated identifiers accepted.',results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    model.close()

if __name__=='__main__':main()
