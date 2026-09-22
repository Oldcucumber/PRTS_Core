"""Compare target-independent LED-only versus general route-sign reading.

Development regression, not an independent holdout. Freeze pixels before runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from prts_core.native_vlm import NativeVLM
from prts_core.ocr import LocalOCR
from prts_core.tasks import identifiers
from prts_core.vehicle_evidence import read_vehicle
import re

PROMPTS={
    'screen':'这辆公交车线路显示屏上的线路号码是什么？只回答线路标识或无法确定。车牌、车身编号、座位数都不是线路。',
    'signage':'识别这辆公交车当前服务的完整线路标识。线路牌可以是电子屏、印刷牌或纸牌；不是车牌、车身资产编号、广告或载客数。只回答线路标识；无法区分则回答无法确定。'
}


def prepare(out):
    source=ROOT/'outputs/stage2/native-vlm/quantization-v2'
    frozen=json.loads((source/'cases.json').read_text(encoding='utf-8'))
    cases=[]
    for case in frozen['cases']:
        if not case.get('image'):continue
        shutil.copyfile(source/case['image'],out/case['image']);cases.append(case)
    run=ROOT/'outputs/stage2/streaming/bus-dml-v1'
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    events=[json.loads(s) for s in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    start=next(e['observation_s'] for e in events if e['type']=='transcript')
    record=next(e for e in events if e['type']=='target_evidence' and
                any(x['text']=='912' for x in e['ocr']))
    case=summary['scenario'];stamp=case['start']+record['observation_s']-start
    cap=cv2.VideoCapture(case['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    cap.set(cv2.CAP_PROP_POS_MSEC,stamp*1000);ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('Missing original private video frame')
    h,w=frame.shape[:2];x1,y1,x2,y2=case['roi'];frame=frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]
    detection=next(d for d in record['detections'] if d['label']=='bus')
    h,w=frame.shape[:2];x1,y1,x2,y2=map(int,np.asarray(detection['box'])*[w,h,w,h])
    file=out/'printed-front.png';cv2.imwrite(str(file),frame[y1:y2,x1:x2])
    cases.append(dict(id='printed-front',image=file.name,expected='912',forbidden=['12','80'],
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),source=case['source'],source_s=stamp,
        box=detection['box'],private=True))
    (out/'cases.json').write_text(json.dumps(dict(scope='Development, not heldout',prompts=PROMPTS,cases=cases),ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--variant',choices=list(PROMPTS))
    ap.add_argument('--language',choices=['Q4_K_M','Q5_K_M','Q8_0','F16'],default='Q4_K_M')
    ap.add_argument('--ocr-profile',choices=['legacy','bounded_upright'],default='legacy')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.prepare:
        if (a.output/'cases.json').exists():raise FileExistsError('Preserve frozen cases')
        prepare(a.output);return
    if not a.variant:ap.error('Choose a prompt variant')
    frozen=json.loads((a.output/'cases.json').read_text(encoding='utf-8'));rows=[]
    tag=a.variant if a.language=='Q4_K_M' else a.variant+'-'+a.language
    if a.ocr_profile!='legacy':tag+='-'+a.ocr_profile
    if (a.output/f'{tag}.json').exists():raise FileExistsError('Preserve previous results')
    log=(a.output/f'{tag}.log').open('w');os.dup2(log.fileno(),2)
    model=NativeVLM(ROOT/'models',ROOT/'outputs/prts-native-build/libprts_vlm.dll',image_slices=1,
                    language_file=f'MiniCPM-V-4_6-{a.language}.gguf');ocr=LocalOCR(profile=a.ocr_profile)
    try:
        for case in frozen['cases']:
            p=a.output/case['image'];assert hashlib.sha256(p.read_bytes()).hexdigest()==case['sha256']
            image=cv2.imread(str(p));result=None
            def verify(crop):
                nonlocal result
                result=model.generate(frozen['prompts'][a.variant],crop,max_tokens=40,raw_prompt=True)
                ids=identifiers(result['text'])
                clear=len(set(ids))==1 and not re.search('无法|不确定|可能|看不清|没有|未能',result['text']) and result['finish_reason']=='stop'
                return dict(route_id=ids[0] if clear else '',raw_answer=result['text'],method=a.variant)
            entries=read_vehicle(image,dict(box=[0,0,1,1],label='bus',observation_id='vehicle'),ocr,verify)
            verified=list(dict.fromkeys(i for e in entries if e.get('route_verified') for i in identifiers(e['text'])
                                        if i==e['route_verification']['route_id']))
            row=dict(id=case['id'],vlm=result,verified=verified,ocr=entries,
                     passed=(case['expected'] in verified if case['expected'] else not verified)
                            and not any(i in verified for i in case['forbidden']))
            rows.append(row);print(json.dumps(dict(id=row['id'],text=result['text'] if result else None,verified=verified,passed=row['passed']),ensure_ascii=False),flush=True)
    finally:model.close()
    (a.output/f'{tag}.json').write_text(json.dumps(dict(variant=a.variant,language=a.language,rows=rows,passed=all(r['passed'] for r in rows)),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
