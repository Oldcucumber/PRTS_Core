"""Compare native pixel-gap splitting with the Python reference on frozen OCR rows."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from prts_core.queue_evidence import split_numeric_row

def main():
    output=ROOT/'outputs/stage2/queue-holdout/native-split-v1'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    path=ROOT/'outputs/native-toolchain/prts_queue.dll';lib=C.CDLL(str(path))
    pb=C.POINTER(C.c_uint8);pf=C.POINTER(C.c_float);pi=C.POINTER(C.c_int32)
    lib.prts_queue_split_rgb.argtypes=[pb,C.c_int32,C.c_int32,pf,C.c_int32,pi,pf]
    lib.prts_queue_split_rgb.restype=C.c_int32
    inputs=[]
    for name in ['first-pass-v1','first-pass-v2','layout-dev-v3']:
        report=json.loads((ROOT/f'outputs/stage2/queue-holdout/{name}/results.json').read_text(encoding='utf-8'))
        for case in report['results']:
            source=cv2.imread(str(ROOT/case['case']['source']))
            for e in case['ocr']:
                if re.fullmatch(r'[0-9]{3,}',e['text']):inputs.append((name+'/'+case['case']['id'],source,e))
    for parts in [['734','18','9502'],['29','6301','850'],['012345678']]:
        positions=[];x=3;text=''.join(parts)
        for group in parts:
            for c in group:positions.append(x);x+=12
            x+=30
        source=np.zeros((42,x+10,3),np.uint8)
        for x0 in positions:cv2.rectangle(source,(x0,5),(x0+7,33),(255,255,255),-1)
        e=dict(text=text,score=1.,box=[[0,0],[x,0],[x,39],[0,39]])
        inputs.append(('synthetic/'+','.join(parts),source,e))
    rows=[]
    for name,image,e in inputs:
        rgb=np.ascontiguousarray(image[:,:,::-1]);h,w=image.shape[:2]
        q=np.asarray(e['box'],np.float32);count=len(e['text'])
        ranges=np.empty((count,2),np.int32);boxes=np.empty((count,4,2),np.float32)
        n=lib.prts_queue_split_rgb(rgb.ctypes.data_as(pb),w,h,q.ctypes.data_as(pf),count,
                                   ranges.ctypes.data_as(pi),boxes.ctypes.data_as(pf))
        if n<1:raise RuntimeError('Native split rejected input')
        expected=split_numeric_row(image,e);actual=[e['text'][a:b] for a,b in ranges[:n]]
        error=float(np.max(np.abs(np.asarray([p['box'] for p in expected])-boxes[:n]))) if len(expected)==n else None
        rows.append(dict(source=name,ocr=e['text'],expected=[p['text'] for p in expected],actual=actual,
                         box_max_error_px=error,passed=actual==[p['text'] for p in expected] and error<=1.0))
    report=dict(passed=all(r['passed'] for r in rows),cases=rows,scope='Same images/OCR boxes, numeric C++/Python parity only; Windows x64 build, no Apple or new OCR quality claim.',
                sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,ROOT/'native/portable/prts_queue.cpp',ROOT/'prts_core/queue_evidence.py']})
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=report['passed'],rows=len(rows),failures=[r for r in rows if not r['passed']]),ensure_ascii=False))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
