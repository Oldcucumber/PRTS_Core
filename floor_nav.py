"""CPU-friendly floor segmentation and image-space corridor prototype."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL = 'nvidia/segformer-b0-finetuned-ade-512-512'


def corridor(mask):
    """Keep near-field floor; derive connected, inset row centers (not metric clearance)."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    seeds = labels[int(h*.85):, int(w*.2):int(w*.8)]
    counts = np.bincount(seeds.ravel(), minlength=n)
    counts[0] = 0
    if not counts.any():
        return np.zeros_like(mask), []
    region = (labels == counts.argmax()).astype(np.uint8)
    inset = cv2.erode(region, np.ones((7, 7), np.uint8))
    points = []
    prev = w/2
    for y in range(int(h*.94), int(h*.25), -max(1, h//35)):
        row = inset[y].astype(np.int16)
        edges = np.diff(np.r_[0, row, 0])
        runs = [(a,b) for a,b in zip(np.where(edges==1)[0], np.where(edges==-1)[0]) if b-a >= w*.08]
        if not runs:
            break
        a,b = min(runs, key=lambda ab: abs((ab[0]+ab[1])/2-prev))
        x = int((a+b)/2)
        if points:
            # A centerline must not bridge an intervening obstacle.
            probe = np.zeros_like(region)
            cv2.line(probe, points[-1], (x,y), 1, 1)
            if np.any((probe != 0) & (inset == 0)):
                break
        points.append((x,y))
        prev = x
    return region.astype(bool), points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, default=Path('outputs'))
    ap.add_argument('--fps', type=float, default=8)
    ap.add_argument('--height', type=int, default=640)
    ap.add_argument('--threshold', type=float, default=.55)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.fps <= 0 or args.height < 128 or not 0 < args.threshold < 1:
        ap.error('fps > 0, height >= 128, and 0 < threshold < 1 required')
    torch.set_num_threads(args.threads)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.input}')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise RuntimeError('Input has invalid FPS')
    stride = max(1, round(source_fps / args.fps))
    fps = source_fps / stride
    args.output.mkdir(parents=True, exist_ok=True)
    processor = SegformerImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
    floor_id = next(k for k,v in model.config.id2label.items() if v.strip()=='floor')
    writer = None
    records, timings, snapshots = [], [], []
    index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            h = args.height
            w = round(frame.shape[1]*h/frame.shape[0]/2)*2
            frame = cv2.resize(frame,(w,h))
            t = time.perf_counter()
            inputs = processor(images=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), size={'height':512,'width':max(32,round(w/h*512/32)*32)}, return_tensors='pt')
            with torch.inference_mode():
                logits = model(**inputs).logits
                prob = logits.softmax(1)
                floor = torch.nn.functional.interpolate(prob[:,floor_id:floor_id+1], size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
                classes = torch.nn.functional.interpolate(logits, size=(h,w),mode='bilinear',align_corners=False).argmax(1)[0].numpy()
            raw = (floor >= args.threshold) & (classes == floor_id)
            region, points = corridor(raw)
            # Use current-frame geometry: never average a path through new obstacles.
            target = points[min(len(points)-1, max(1,int(len(points)*.7)))] if len(points)>=5 else None
            direction = 'UNKNOWN'
            offset = None
            if target is not None:
                offset = (target[0]-w/2)/(w/2)
                direction = 'LEFT' if offset < -.15 else 'RIGHT' if offset > .15 else 'FORWARD'
            elapsed = time.perf_counter()-t
            timings.append(elapsed)
            display = frame.copy()
            tint = np.full_like(display,(60,200,40))
            display[region] = (display[region]*.65+tint[region]*.35).astype(np.uint8)
            contours,_ = cv2.findContours(region.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display,contours,-1,(70,240,90),1)
            if len(points)>1:
                cv2.polylines(display,[np.array(points,np.int32)],False,(0,220,255),3)
            if target:
                cv2.arrowedLine(display,points[0],target,(255,160,20),3,tipLength=.15)
            cv2.rectangle(display,(0,0),(w,91),(15,20,25),-1)
            for k,line in enumerate([f'Candidate: {direction}', f't={index/source_fps:.2f}s | infer+path {elapsed*1000:.0f}ms', 'Green: floor | yellow: image-space path', 'Prototype: no distance / safety guarantee']):
                cv2.putText(display,line,(8,20+k*20),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(args.output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create output video')
            writer.write(display)
            records.append({'frame':index,'time_s':index/source_fps,'direction':direction,'horizontal_offset':offset,'floor_fraction':float(region.mean()),'centerline':points,'processing_ms':elapsed*1000})
            snapshots.append(display)
            print(f'frame {index}: {direction}, floor={region.mean():.2f}, {elapsed*1000:.0f}ms',flush=True)
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not records:
        raise RuntimeError('No frames decoded')
    chosen = np.linspace(0,len(snapshots)-1,min(6,len(snapshots)),dtype=int)
    sheet = np.concatenate([snapshots[i] for i in chosen],axis=1)
    cv2.imwrite(str(args.output/'preview.jpg'),sheet)
    report = {'model':MODEL,'source':str(args.input.resolve()),'source_fps':source_fps,'decoded_frames':index,'processed_frames':len(records),'output_fps':fps,'mean_processing_ms':float(np.mean(timings)*1000),'p95_processing_ms':float(np.percentile(timings,95)*1000),'processing_fps':1/float(np.mean(timings)),'wall_seconds':time.perf_counter()-started,'frames':records}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))

if __name__ == '__main__':
    main()
