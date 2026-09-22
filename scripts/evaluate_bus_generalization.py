"""Real-video bus OCR/association audit; first-pass outputs must be preserved.

Runs the production detector, OCR and Tasks matcher without segmentation, which
does not participate in route recognition. This is sampled offline evaluation,
not a wall-clock response or iPhone benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from prts_core.ocr import LocalOCR
from prts_core.tasks import Tasks
from prts_core.vehicle_evidence import read_vehicle,verify_with_vlm
from ultralytics import YOLO


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fps',type=float,default=2)
    ap.add_argument('--case',action='append')
    ap.add_argument('--spec',type=Path,default=ROOT/'tests/fixtures/bus_generalization.json')
    ap.add_argument('--native-library',type=Path)
    ap.add_argument('--language',default='Q5_K_M')
    ap.add_argument('--projector',default='mmproj-model-f16.gguf')
    ap.add_argument('--image-slices',type=int,default=1)
    args=ap.parse_args()
    if (args.output/'results.json').exists():
        raise FileExistsError('Use a new output directory to preserve the previous evaluation.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'run-arguments.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8')
    spec_bytes=args.spec.read_bytes()
    spec=json.loads(spec_bytes)
    (args.output/'spec_snapshot.json').write_bytes(spec_bytes)
    frozen=[]
    for case in spec['cases']:
        if args.case and case['id'] not in args.case:continue
        source=Path(case['source']) if case.get('source') else ROOT/'outputs/stage2/bus-generalization/data'/(case['id']+Path(case['file']).suffix)
        with source.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        frozen.append(dict(id=case['id'],source=str(source),bytes=source.stat().st_size,sha256=sha))
    (args.output/'input-manifest.json').write_text(json.dumps(dict(spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),inputs=frozen),ensure_ascii=False,indent=2),encoding='utf-8')
    log=(args.output/'native-stderr.log').open('w');os.dup2(log.fileno(),2)
    tasks_source=(ROOT/'prts_core/tasks.py').read_bytes()
    (args.output/'tasks_snapshot.py').write_bytes(tasks_source)
    tasks_sha256=hashlib.sha256(tasks_source).hexdigest()
    snapshots={}
    for name in ('tasks.py','vehicle_evidence.py','ocr.py','native_vlm.py'):
        data=(ROOT/'prts_core'/name).read_bytes();snapshots[name]=hashlib.sha256(data).hexdigest()
        (args.output/(Path(name).stem+'_snapshot.py')).write_bytes(data)
    torch.set_num_threads(4)
    det=YOLO(str(ROOT/'models/yolo11n.pt'));ocr=LocalOCR()
    verifier=None
    if args.native_library:
        from prts_core.native_vlm import NativeVLM
        native=NativeVLM(ROOT/'models',args.native_library,image_slices=args.image_slices,
                         language_file=f'MiniCPM-V-4_6-{args.language}.gguf',projector_file=args.projector)
        verifier=lambda crop:verify_with_vlm(native,crop)
    results=[]
    for case in spec['cases']:
        if args.case and case['id'] not in args.case:continue
        source=Path(case['source']) if case.get('source') else ROOT/'outputs/stage2/bus-generalization/data'/(case['id']+Path(case['file']).suffix)
        static=case.get('media_type')=='image'
        cap=None if static else cv2.VideoCapture(str(source))
        if not static and not cap.isOpened():raise FileNotFoundError(source)
        duration=1/args.fps if static else cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
        waiting={}
        for target in ([case['expected_route']] if case['expected_route'] is not None else [])+case['negative_targets']:
            task=Tasks();task.command(dict(action='wait',kind='bus',target=target),0);waiting[target]=task
        events=[];samples=[]
        for seq,t in enumerate(np.arange(case.get('start',0),min(duration,case.get('end',duration)),1/args.fps)):
            if static:frame=cv2.imread(str(source));ok=frame is not None
            else:cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
            if not ok:continue
            if case.get('roi'):
                h,w=frame.shape[:2];x1,y1,x2,y2=case['roi'];frame=frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)].copy()
            for x1,y1,x2,y2 in case.get('ignore_regions',[]):
                h,w=frame.shape[:2];frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]=0
            start=time.perf_counter();h,w=frame.shape[:2]
            predictions=det.predict(frame,imgsz=640,conf=.25,iou=.55,device='cpu',verbose=False)[0]
            detections=[];entries=[]
            for i,box in enumerate(predictions.boxes):
                cid=int(box.cls.item());label=det.names[cid]
                if label not in ('bus','train'):continue
                b=box.xyxyn[0].tolist();d=dict(box=b,label=label,score=float(box.conf.item()),observation_id=f'{seq}:{i}')
                detections.append(d)
                entries.extend(read_vehicle(frame,d,ocr,verifier))
            sample=dict(sequence=seq,source_s=float(t),detections=detections,ocr=entries,processing_ms=(time.perf_counter()-start)*1000)
            samples.append(sample)
            for target,task in waiting.items():
                event=task.observe(entries,detections,float(t))
                if event:
                    events.append(dict(requested_target=target,**event))
                    print(json.dumps(dict(case=case['id'],target=target,source_s=float(t),evidence=event['evidence_text']),ensure_ascii=False),flush=True)
                    cv2.imwrite(str(args.output/f'{case["id"]}-{target}-{seq}.jpg'),frame)
            if seq%10==0:print(json.dumps(dict(case=case['id'],frames=seq+1)),flush=True)
        if cap:cap.release()
        result=dict(case=case,source=str(source),sample_fps=args.fps,samples=len(samples),events=events,
                    positive_observed=waiting[case['expected_route']].wait['notified'] if case['expected_route'] is not None else None,
                    false_positive_targets=[target for target in case['negative_targets'] if waiting[target].wait['notified']])
        (args.output/(case['id']+'-evidence.json')).write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
        results.append(result)
        report=dict(results=results,complete=len(results)==len(args.case or spec['cases']),
                    detector='yolo11n.pt',ocr='RapidOCR PP-OCRv6-small',
                    tasks_sha256=tasks_sha256,
                    implementation_sha256=snapshots,language=args.language,ocr_profile='bounded_upright',
                    route_verifier='native_vlm+ocr' if verifier else None,
                    status='sampled offline real video; not a latency or deployment estimate')
        (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    if args.native_library:native.close()


if __name__=='__main__':main()
