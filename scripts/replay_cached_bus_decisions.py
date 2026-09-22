"""Run current task rules on frozen neural evidence; this is not new inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prts_core.tasks import Tasks

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[]
    for name in ['vision-q8-regression-v1','vision-q8-holdout-v3']:
        folder=ROOT/'outputs/stage2/bus-generalization'/name
        report=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        for case in report['results']:
            c=case['case'];file=folder/(c['id']+'-evidence.json')
            evidence=json.loads(file.read_text(encoding='utf-8'));events=[]
            targets=([c['expected_route']] if c['expected_route'] else [])+c['negative_targets']
            for target in targets:
                state=Tasks();state.command(dict(action='wait',kind='bus',target=target),0)
                for frame in evidence:
                    event=state.observe(frame['ocr'],frame['detections'],frame['source_s'])
                    if event and event['type']=='target_observed':events.append(dict(target=target,event=event))
            found=any(e['target']==c['expected_route'] for e in events) if c['expected_route'] else None
            false=[e['target'] for e in events if e['target'] in c['negative_targets']]
            rows.append(dict(set=name,id=c['id'],positive_observed=found,false_positive_targets=false,
                unchanged=found==case['positive_observed'] and false==case['false_positive_targets'],frames=len(evidence),
                evidence_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    args.output.mkdir(parents=True)
    result=dict(scope=__doc__,results=rows,passed=all(r['unchanged'] for r in rows),
                tasks_sha256=hashlib.sha256((ROOT/'prts_core/tasks.py').read_bytes()).hexdigest())
    (args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=result['passed'],cases=len(rows),frames=sum(r['frames'] for r in rows))))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
