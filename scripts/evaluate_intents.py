"""Run actual local rule/model routing on text cases; not an ASR evaluation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prts_core.intent import route
from prts_core.models import Brain

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('outputs/intent-regression.json'));args=ap.parse_args()
    cases=json.loads(args.cases.read_text(encoding='utf8'))['cases'];results=[];brain=None
    for case in cases:
        started=time.perf_counter();actual=route(case['text'],case.get('current_wait'))
        if actual is None:
            if brain is None:brain=Brain()
            actual=brain.intent(case['text']);actual['source']='local_language_model'
        passed=all(actual.get(k,'')==v for k,v in case['expected'].items())
        results.append(dict(**case,actual=actual,passed=passed,processing_ms=(time.perf_counter()-started)*1000))
    report=dict(provenance='Synthetic text regression; initial independent test was 17/18 before corrections; not audio accuracy',
                passed=sum(r['passed'] for r in results),total=len(results),cases=results)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(f"{report['passed']}/{report['total']}")
    if report['passed']!=report['total']:raise SystemExit(1)

if __name__=='__main__':main()
