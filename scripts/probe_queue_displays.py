"""Diagnostic screen crops from actual detector boxes, not hand-picked target ROIs."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import torch
from ultralytics import YOLO
from prts_core.ocr import LocalOCR
from prts_core.native_vlm import NativeVLM

PROMPT=('读取这个显示屏中已经叫到或已经可取餐的完整号码，保留字母、前导零和后缀。'
        '不要报柜台编号、等候号码、价格、汇率或广告数字。仅返回号码，以逗号分隔。'
        '模糊或没有叫号时返回“无法确定”。')


def main():
    out=ROOT/'outputs/stage2/queue-holdout/display-probe-v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    log=(out/'native-stderr.log').open('w');os.dup2(log.fileno(),2)
    (out/'prompt.txt').write_text(PROMPT,encoding='utf-8')
    (out/'runner.py.snapshot').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);detector=YOLO(str(ROOT/'models/yolo11n.pt'));ocr=LocalOCR()
    model=NativeVLM(ROOT/'models',ROOT/'outputs/prts-native-build/libprts_vlm.dll',image_slices=1,
                    language_file='MiniCPM-V-4_6-Q5_K_M.gguf',projector_file='mmproj-matrix-q8_0.gguf')
    rows=[]
    for case in json.loads((ROOT/'tests/fixtures/queue_holdout_v1.json').read_text(encoding='utf-8'))['cases']:
        frame=cv2.imread(str(ROOT/case['source']));h,w=frame.shape[:2]
        boxes=detector.predict(frame,imgsz=640,conf=.25,iou=.55,device='cpu',verbose=False)[0].boxes
        screens=[]
        for index,box in enumerate(boxes):
            label=detector.names[int(box.cls.item())]
            if label not in ('tv','laptop'):continue
            x1,y1,x2,y2=[round(float(v)) for v in box.xyxy[0]]
            crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)].copy()
            name=f'{case["id"]}-{index}.jpg';cv2.imwrite(str(out/name),crop)
            entries=ocr(crop);response=model.generate(PROMPT,crop,max_tokens=120,raw_prompt=True)
            screens.append(dict(box=[x1,y1,x2,y2],label=label,score=float(box.conf.item()),crop=name,ocr=entries,vlm=response))
        row=dict(case=case['id'],screens=screens);rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(status='Development diagnosis only; detector crops and target-blind VLM, not integrated or a new holdout',results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(case=case['id'],screens=[dict(crop=s['crop'],ocr=[e['text'] for e in s['ocr']],vlm=s['vlm']['text']) for s in screens]),ensure_ascii=False),flush=True)
    model.close()


if __name__=='__main__':main()
