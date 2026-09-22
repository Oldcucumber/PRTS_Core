"""Development-only, target-independent crop reading. Preserve the failed holdout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from prts_core.native_vlm import NativeVLM
from prts_core.ocr import LocalOCR
from prts_core.vehicle_evidence import verify_with_vlm
from prts_core.tasks import identifiers


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'results.json').exists():raise FileExistsError('Preserve the earlier probe')
    log=(a.output/'native.log').open('w');os.dup2(log.fileno(),2)
    cases=[];data=ROOT/'outputs/stage2/bus-generalization'
    sample=json.loads((data/'holdout-v3-first-pass/commons-nwfb-101x-evidence.json').read_text(encoding='utf-8'))[0]
    image=cv2.imread(str(data/'data/commons-nwfb-101x.JPG'));h,w=image.shape[:2]
    x1,y1,x2,y2=map(int,np.asarray(sample['detections'][0]['box'])*[w,h,w,h])
    cases.append(('101x-development',image[y1:y2,x1:x2]))
    base=data/'sign-reading-v3'
    for name in ['front-8','rear-fleet','printed-front']:
        meta=json.loads((base/'cases.json').read_text(encoding='utf-8'))
        case=next(c for c in meta['cases'] if c['id']==name)
        cases.append((name,cv2.imread(str(base/case['image']))))
    ocr=LocalOCR();model=NativeVLM(ROOT/'models',ROOT/'outputs/prts-native-build/libprts_vlm.dll',
        image_slices=1,language_file='MiniCPM-V-4_6-Q5_K_M.gguf');rows=[]
    try:
        for name,image in cases:
            h,w=image.shape[:2]
            for view,(x1,y1,x2,y2) in dict(full=(0,0,1,1),upper=(0,0,1,.62),
                    upper_left=(0,0,.6,.62),upper_right=(.4,0,1,.62)).items():
                crop=image[round(h*y1):round(h*y2),round(w*x1):round(w*x2)]
                file=a.output/f'{name}-{view}.png';cv2.imwrite(str(file),crop)
                text=ocr(crop);verification=verify_with_vlm(model,crop)
                verified=bool(verification['route_id'] and any(verification['route_id'] in identifiers(e['text']) and e['score']>=.45 for e in text))
                row=dict(case=name,view=view,ocr=text,verification=verification,confirmed_by_both=verified,
                         image_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                rows.append(row);print(json.dumps(dict(case=name,view=view,text=[e['text'] for e in text],route=verification['route_id'],verified=verified),ensure_ascii=False),flush=True)
    finally:model.close()
    (a.output/'results.json').write_text(json.dumps(dict(scope='Development crop comparison; no expected route is supplied to the models',rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
