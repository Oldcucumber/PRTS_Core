"""Frozen task regression: same native runtime/projector/input, F16 versus Q4 language.

These are development regression cases, not a new independent accuracy estimate.
Preparation freezes pixel hashes and target-independent prompts before inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def prepare(out):
    import cv2
    import numpy as np
    folder=ROOT/'outputs/stage2/bus-generalization'
    prompt='这辆公交车线路显示屏上的线路号码是什么？只回答线路标识或无法确定。车牌、车身编号、座位数都不是线路。'
    rows=[]
    specs=[('front-8','commons-utrecht-8','native-dev-utrecht',0.,'0:0','8'),
           ('rear-fleet','commons-utrecht-8','native-dev-utrecht',2.5,'5:0',None),
           ('route-20','commons-citybus-20','native-dev',0.,'0:0','20'),
           ('suffix-20A','commons-citybus-20a','holdout-v2-native',0.,'0:0','20A'),
           ('prefix-SL3','commons-mbta-sl3','holdout-v2-native',0.,'0:0','SL3')]
    for name,case,run,stamp,vehicle,expected in specs:
        records=json.loads((folder/run/f'{case}-evidence.json').read_text(encoding='utf-8'))
        row=next(r for r in records if r['source_s']==stamp)
        detection=next(d for d in row['detections'] if d['observation_id']==vehicle)
        # Exact source filename, never a derived contact sheet sharing the stem.
        sources=[]
        for manifest in ('sources.json','holdout-v2-sources.json'):
            sources.extend(json.loads((folder/'data'/manifest).read_text(encoding='utf-8')))
        source_record=next(s for s in sources if s['id']==case)
        source=ROOT/source_record['local_file']
        if source.suffix=='.jpg':bgr=cv2.imread(str(source))
        else:
            cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
            cap.set(cv2.CAP_PROP_POS_MSEC,stamp*1000);ok,bgr=cap.read();cap.release()
            if not ok:raise RuntimeError(source)
        h,w=bgr.shape[:2];x1,y1,x2,y2=map(int,np.asarray(detection['box'])*[w,h,w,h])
        image=out/f'{name}.png';cv2.imwrite(str(image),bgr[max(0,y1):min(h,y2),max(0,x1):min(w,x2)])
        rows.append(dict(id=name,image=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                         source=str(source.relative_to(ROOT)),source_s=stamp,box=detection['box'],prompt=prompt,
                         expected=expected,forbidden=['4833'] if name=='rear-fleet' else []))
    rows.extend([
        dict(id='suffix-intent',prompt='从“请帮我等57B路公交车”提取完整线路标识，只输出标识，不省略字母。',expected='57B',forbidden=[]),
        dict(id='prefix-intent',prompt='从“请帮我等SL3路公交车”提取完整线路标识，只输出标识，不省略字母。',expected='SL3',forbidden=[]),
        dict(id='queue-intent',prompt='从“等B205号叫号时提醒我”提取完整叫号标识，只输出标识。',expected='B205',forbidden=[])
    ])
    (out/'cases.json').write_text(json.dumps(dict(scope='Development task regression, not heldout',
        gate='All exact identifier cases correct; fleet number never emitted as route; finish_reason stop.',cases=rows),ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--language',choices=['Q4_K_M','Q5_K_M','Q8_0','F16'])
    ap.add_argument('--native-library',type=Path,default=ROOT/'outputs/prts-native-build/libprts_vlm.dll')
    ap.add_argument('--projector',default='mmproj-model-f16.gguf')
    ap.add_argument('--image-slices',type=int,default=1)
    ap.add_argument('--cases',type=Path,help='Reuse an existing frozen input directory without copying or altering it')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/stage2/native-vlm/quantization')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        if (args.output/'cases.json').exists():raise FileExistsError('Keep the frozen regression inputs')
        prepare(args.output);return
    if not args.language:ap.error('Choose --prepare or --language')
    import cv2
    import psutil
    from prts_core.native_vlm import NativeVLM
    from prts_core.tasks import identifiers
    if (args.output/f'{args.language}.json').exists():raise FileExistsError('Preserve the previous comparison')
    case_folder=args.cases or args.output
    frozen=json.loads((case_folder/'cases.json').read_text(encoding='utf-8'));rows=[]
    log=(args.output/f'{args.language}-stderr.log').open('w');os.dup2(log.fileno(),2)
    process=psutil.Process();done=threading.Event();memory=[]
    def sample():
        while not done.wait(.05):
            info=process.memory_info();memory.append((info.rss,getattr(info,'peak_wset',info.rss)))
    monitor=threading.Thread(target=sample,daemon=True);monitor.start();started=time.perf_counter()
    model=NativeVLM(ROOT/'models',args.native_library,image_slices=args.image_slices,
                    language_file=f'MiniCPM-V-4_6-{args.language}.gguf',projector_file=args.projector)
    load_s=time.perf_counter()-started
    try:
        for case in frozen['cases']:
            image=None
            if case.get('image'):
                p=case_folder/case['image'];assert hashlib.sha256(p.read_bytes()).hexdigest()==case['sha256']
                image=cv2.imread(str(p))
            deltas=[];start=time.perf_counter()
            result=model.generate(case['prompt'],image,raw_prompt=True,max_tokens=60,
                                  on_delta=lambda text:deltas.append((time.perf_counter()-start,text)))
            ids=identifiers(result['text'])
            exact=case['expected'] is None or ids==[case['expected']]
            passed=exact and not any(i in ids for i in case['forbidden']) and result['finish_reason']=='stop'
            rows.append(dict(case=case['id'],**result,passed=passed,first_delta_ms=deltas[0][0]*1000 if deltas else None))
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    finally:model.close();done.set();monitor.join()
    report=dict(language=args.language,image_slices=args.image_slices,projector=args.projector,load_s=load_s,
                projector_sha256=hashlib.sha256((ROOT/'models/minicpm-v-4.6-gguf'/args.projector).read_bytes()).hexdigest(),
                peak_sampled_rss=max(x[0] for x in memory),peak_working_set=max(x[1] for x in memory),
                passed=all(r['passed'] for r in rows),results=rows,
                native_library=str(args.native_library),native_library_sha256=hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
                cases_sha256=hashlib.sha256((case_folder/'cases.json').read_bytes()).hexdigest())
    (args.output/f'{args.language}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
