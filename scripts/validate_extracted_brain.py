"""Verify a relocated bundle and execute its packaged models and recorded input."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    profile=json.loads((root/'BUNDLE.json').read_text(encoding='utf-8'))['runtime']
    results=[]
    def run(label,arguments):
        start=time.monotonic()
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,'-X','utf8',*arguments],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(name=label,exit_code=p.returncode,seconds=time.monotonic()-start))
        if p.returncode:raise RuntimeError('Relocation check failed: '+label)
    run('hash-verification',['scripts/run_bundle_replay.py','--verify-only','--output',str(out/'verify')])
    run('unit-tests',['-m','unittest','discover','-s','tests','-q'])
    run('real-brain',['scripts/validate_brain_scenarios.py','--output',str(out/'real-brain'),
        '--only','bank-ticket','bank-counter','--library',str(root/profile['windows_vlm_library']),
        '--language',profile['language_file'],'--projector',Path(profile['projector']).name,
        '--model-subdir',profile['model_subdir'],'--sampling',profile['sampling'],
        '--threads',str(profile['native_threads'])])
    actual=json.loads((out/'real-brain/results.json').read_text(encoding='utf-8'))
    if not actual['complete'] or actual['passed']!=2:raise RuntimeError('Relocated visual brain differs')
    run('real-recording',['scripts/run_bundle_replay.py','--case','notice','--provider','dml','--tts',
        '--output',str(out/'notice')])
    events=[json.loads(s) for s in (out/'notice/events.jsonl').read_text(encoding='utf-8').splitlines()]
    types={e['type'] for e in events}
    if not {'scene','transcript','answer'}<=types or types&{'request_error','worker_error'}:
        raise RuntimeError('Relocated recorded interaction incomplete')
    result=dict(complete=True,bundle_root=str(root),python=sys.executable,checks=results,
        brain_cases_passed=actual['passed'],recorded_input_summary=json.loads((out/'notice/summary.json').read_text(encoding='utf-8')),
        scope='Extracted code, weights and media; existing installed Windows Python dependencies. Not a fresh OS or Apple execution.')
    (out/'extraction-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,root=str(root),checks=len(results))))

if __name__=='__main__':main()
