"""Actual visual-model checks, fixed paired targets and temporal context before inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from prts_core.native_vlm import NativeVLM
from prts_core.waiting_brain import WaitingBrain
from prts_core.ocr import LocalOCR


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--language',default='MiniCPM-V-4_6-Q5_K_M.gguf')
    ap.add_argument('--projector',default='mmproj-matrix-q8_0.gguf')
    ap.add_argument('--model-subdir',default='minicpm-v-4.6-gguf')
    ap.add_argument('--server',type=Path)
    ap.add_argument('--strategy',choices=['direct','scene_agent'],default='direct')
    ap.add_argument('--only',nargs='+',help='Explicit bounded diagnostic subset; results keep subset scope')
    ap.add_argument('--spec',type=Path,help='Previously frozen additional cases')
    ap.add_argument('--sampling',choices=['greedy','recommended'],default='greedy')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--library',type=Path,default=ROOT/'outputs/prts-native-build/libprts_vlm.dll')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    config=dict(vars(args))
    if not args.server:
        with args.library.open('rb') as f:config['native_library_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    (out/'run-arguments.json').write_text(json.dumps(config,default=str,indent=2),encoding='utf-8')
    qdir='outputs/stage2/queue-holdout/'
    bdir='outputs/stage2/bus-generalization/data/'
    cases=[]
    def case(id,source,kind,target,expected,**extra):
        cases.append(dict(id=id,source=source,kind=kind,target=target,expected=expected,**extra))
    for id,file,target,expected in [
        ('bus101','commons-manchester-101.jpg','101',True),
        ('bus101-not912','commons-manchester-101.jpg','912',False),
        ('fleet-not-route','commons-manchester-101.jpg','17663',False),
        ('suffix101x','commons-nwfb-101x.JPG','101X',True),
        ('suffix-not101','commons-nwfb-101x.JPG','101',False),
        ('stop-not-arrival','commons-k1-stop.JPG','K1',False)]:case(id,bdir+file,'bus',target,expected)
    for id,file,target,expected in [
        ('bank-ticket','data/commons-bank-queue.jpg','C002',True),
        ('bank-counter','data/commons-bank-queue.jpg','3',False),
        ('food-ready','data/commons-hk-foodcourt.jpg','215',True),
        ('food-other','data/commons-hk-foodcourt.jpg','194',False),
        ('hospital-ticket','data-v2/masmitra.jpg','B003',True),
        ('hospital-counter','data-v2/masmitra.jpg','2',False),
        ('led-ticket','data-v2/japan-led.jpg','555',True),
        ('led-counter','data-v2/japan-led.jpg','8',False),
        ('pharmacy-ticket','data-v2/pharmacy.jpg','43',True),
        ('pharmacy-other','data-v2/pharmacy.jpg','44',False),
        ('price-not-ticket','data-v2/burrito-ad.jpg','350',False)]:case(id,qdir+file,'number',target,expected)
    # Same task sees a price sign then a real ticket display. History is actual model output.
    case('temporal-before',qdir+'data-v2/burrito-ad.jpg','number','B003',False,version=100)
    case('temporal-after',qdir+'data-v2/masmitra.jpg','number','B003',True,version=100)
    for i,(text,expected) in enumerate([('请A108号到3号窗口',True),('请A180号到3号窗口',False),
                                      ('A108还没叫到，请继续等候',False)]):
        case('ambient-'+str(i),qdir+'data-v2/burrito-ad.jpg','number','A108',expected,
             ambient=[dict(text=text,observation_s=0)],annotation='Scripted environment transcript; not real audio recognition.')
    if args.spec:cases=json.loads(args.spec.read_text(encoding='utf-8'))['cases']
    if args.only:cases=[c for c in cases if c['id'] in args.only]
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    for file in [Path(__file__),ROOT/'prts_core/waiting_brain.py',ROOT/'prts_core/native_vlm.py']:
        (out/file.name).write_bytes(file.read_bytes())
    log=(out/'native.log').open('w');os.dup2(log.fileno(),2)
    if args.server:
        from prts_core.server_vlm import ServerVLM
        model=ServerVLM(args.server,ROOT/'models'/args.model_subdir/args.language,
                        ROOT/'models'/args.model_subdir/args.projector,out/'server.log',sampling=args.sampling)
    else:
        model=NativeVLM(ROOT/'models',args.library,image_slices=1,threads=args.threads,
            language_file=args.language,projector_file=args.projector,model_subdir=args.model_subdir,sampling=args.sampling)
    brain=WaitingBrain(model,strategy=args.strategy);ocr=LocalOCR(profile='bounded_upright');cache={};results=[]
    for i,c in enumerate(cases):
        source=ROOT/c['source'];bgr=cv2.imread(str(source))
        if c['source'] not in cache:cache[c['source']]=ocr(bgr)
        task=dict(kind=c['kind'],target=c['target'],direction=c.get('direction',''),version=c.get('version',i+1))
        result=brain.observe(bgr,task,i,cache[c['source']],c.get('ambient',[]))
        passed=(result['decision']=='MATCH')==c['expected']
        results.append(dict(case=c,response=result,passed=passed,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        summary=dict(complete=len(results)==len(cases),passed=sum(x['passed'] for x in results),total=len(results),
            positive_hits=sum(x['case']['expected'] and x['response']['decision']=='MATCH' for x in results),
            positive_count=sum(x['case']['expected'] for x in results),
            false_positives=sum(not x['case']['expected'] and x['response']['decision']=='MATCH' for x in results),
            uncertain=sum(x['response']['decision']=='UNCLEAR' for x in results),
            scope='Real photos, fresh raw OCR and multimodal inference; previously inspected development media. Temporal input is a photo sequence, environment speech is scripted text.',results=results)
        (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(case=c['id'],passed=passed,response=result['model_response']),ensure_ascii=False),flush=True)
    model.close()

if __name__=='__main__':main()
