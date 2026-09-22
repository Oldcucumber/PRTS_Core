"""Small frozen paired-target experiment of the actual multimodal waiting brain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from prts_core.native_vlm import NativeVLM
from prts_core.waiting_brain import WaitingBrain, SYSTEM


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--language',default='MiniCPM-V-4_6-Q5_K_M.gguf')
    ap.add_argument('--projector',default='mmproj-matrix-q8_0.gguf')
    ap.add_argument('--model-subdir',default='minicpm-v-4.6-gguf')
    ap.add_argument('--server',type=Path)
    ap.add_argument('--sampling',choices=['greedy','recommended'],default='greedy')
    ap.add_argument('--library',type=Path,default=ROOT/'outputs/prts-native-build/libprts_vlm.dll')
    args=ap.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False)
    (out/'prompt.txt').write_text(SYSTEM,encoding='utf-8')
    for path in [Path(__file__),ROOT/'prts_core/waiting_brain.py']:
        (out/path.name).write_bytes(path.read_bytes())
    log=(out/'native.log').open('w');os.dup2(log.fileno(),2)
    if args.server:
        from prts_core.server_vlm import ServerVLM
        model=ServerVLM(args.server,ROOT/'models'/args.model_subdir/args.language,
                        ROOT/'models'/args.model_subdir/args.projector,out/'server.log',sampling=args.sampling)
    else:
        model=NativeVLM(ROOT/'models',args.library,image_slices=1,
                        language_file=args.language,projector_file=args.projector,model_subdir=args.model_subdir,sampling=args.sampling)
    brain=WaitingBrain(model);rows=[]
    report=json.loads((ROOT/'outputs/stage2/queue-holdout/layout-dev-v3/results.json').read_text(encoding='utf-8'))
    cases={x['case']['id']:x for x in report['results']}
    # Targets/labels fixed before any call; no expected label in the prompt.
    pairs=[('masmitra','B003',True),('masmitra','2',False),('japan-led','555',True),
           ('japan-led','8',False),('pharmacy','43',True),('pharmacy','44',False),
           ('burrito-ad','350',False),('burrito-ad','3',False)]
    (out/'cases.json').write_text(json.dumps(pairs),encoding='utf-8')
    for index,(name,target,expected) in enumerate(pairs):
        case=cases[name];source=ROOT/case['case']['source'];bgr=cv2.imread(str(source))
        task=dict(kind='number',target=target,direction='',version=index+1)
        response=brain.observe(bgr,task,index,case['ocr'])
        row=dict(case=name,target=target,expected=expected,observed=response['decision']=='MATCH',
                 response=response,source=case['case']['source'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(scope='Frozen paired goals on previously inspected real photos; cached raw OCR only, new multimodal inference.',results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(case=name,target=target,expected=expected,response=response['model_response']),ensure_ascii=False),flush=True)
    model.close()

if __name__=='__main__':main()
