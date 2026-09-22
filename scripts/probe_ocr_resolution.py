"""Compare OCR resize policy on frozen route crops and a real notice frame."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import psutil
from prts_core.ocr import LocalOCR


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--policy',choices=['min','max'],required=True)
    ap.add_argument('--no-classifier',action='store_true')
    a=ap.parse_args();out=ROOT/'outputs/stage2/ocr-resolution';out.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'outputs/stage2/bus-generalization/sign-reading-v3';cases=json.loads((folder/'cases.json').read_text(encoding='utf-8'))['cases']
    ocr=LocalOCR(profile='legacy');ocr.engine.text_det.limit_type=a.policy
    if a.policy=='max':ocr.engine.max_side_len=960
    if a.no_classifier:ocr.engine.use_cls=False
    rows=[]
    for case in cases:
        path=folder/case['image'];image=cv2.imread(str(path));start=time.perf_counter();entries=ocr(image)
        rows.append(dict(id=case['id'],input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ocr=entries,ms=(time.perf_counter()-start)*1000))
    case=json.loads((ROOT/'outputs/stage2/streaming/notice-case.json').read_text(encoding='utf-8'))
    cap=cv2.VideoCapture(case['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);cap.set(cv2.CAP_PROP_POS_MSEC,7000);ok,image=cap.read();cap.release()
    if not ok:raise RuntimeError('Missing private notice source')
    h,w=image.shape[:2];x1,y1,x2,y2=case['roi'];image=image[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]
    start=time.perf_counter();entries=ocr(image);rows.append(dict(id='notice',source=case['source'],source_s=7,ocr=entries,ms=(time.perf_counter()-start)*1000))
    info=psutil.Process().memory_info()
    tag=a.policy+('-upright' if a.no_classifier else '')
    (out/f'{tag}.json').write_text(json.dumps(dict(policy=a.policy,use_cls=not a.no_classifier,rows=rows,rss=info.rss,peak_working_set=info.peak_wset),ensure_ascii=False,indent=2),encoding='utf-8')
    for row in rows:print(row['id'],[(e['text'],round(e['score'],3)) for e in row['ocr']])
    print('peak_working_set',info.peak_wset)


if __name__=='__main__':main()
