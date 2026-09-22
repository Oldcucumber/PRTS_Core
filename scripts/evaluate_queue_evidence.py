"""Freeze and run real queue-photo OCR/task evidence, including absent identifiers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
from prts_core.ocr import LocalOCR
from prts_core.tasks import Tasks
from prts_core.queue_evidence import read_queue


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec',type=Path,default=ROOT/'tests/fixtures/queue_holdout_v1.json')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--layout',action='store_true')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    spec=args.spec.read_bytes();(args.output/'spec_snapshot.json').write_bytes(spec)
    for name in ('ocr','tasks','queue_evidence'):
        (args.output/f'{name}_snapshot.py').write_bytes((ROOT/f'prts_core/{name}.py').read_bytes())
    ocr=LocalOCR();rows=[]
    for case in json.loads(spec)['cases']:
        file=ROOT/case['source']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=case['sha256']:raise ValueError('Input changed')
        image=cv2.imread(str(file));start=time.perf_counter();entries=read_queue(image,ocr) if args.layout else ocr(image)
        observations=[]
        for target in case['positive_targets']+case['negative_targets']:
            tasks=Tasks();tasks.command(dict(action='wait',kind='number',target=target),0)
            event=tasks.observe(entries,[],1)
            observations.append(dict(target=target,canonical_target=tasks.wait['target'],event=event,
                                     positive=target in case['positive_targets']))
        row=dict(case=case,ocr=entries,processing_ms=(time.perf_counter()-start)*1000,observations=observations,
                 misses=[r['target'] for r in observations if r['positive'] and (r['event'] or {}).get('type')!='target_observed'],
                 false_positives=[r['target'] for r in observations if not r['positive'] and (r['event'] or {}).get('type')=='target_observed'],
                 unconfirmed_candidates=[r['target'] for r in observations if (r['event'] or {}).get('type')=='target_candidate'])
        rows.append(row);print(json.dumps({k:row[k] for k in ('misses','false_positives')},ensure_ascii=False),flush=True)
    report=dict(complete=True,layout=args.layout,scope='Real still photographs, OCR and Tasks; no temporal arrival/ASR claims. Previously inspected sources are development data.',results=rows)
    (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
