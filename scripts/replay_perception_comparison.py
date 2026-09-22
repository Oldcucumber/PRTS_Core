"""Factor perception, obstacle fusion and temporal guidance on cached real frames."""
import argparse,collections,importlib.util,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from prts_core.guidance import Guide,draw,validate_path

def load_v1():
    path=Path('outputs/stage2/baseline-v1/guidance.py')
    spec=importlib.util.spec_from_file_location('baseline_guidance',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Guide

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='development');ap.add_argument('--candidate',default='mask2former-fp16')
    ap.add_argument('--cache',type=Path,default=Path('outputs/stage2/perception/comparison'))
    ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/path-comparison'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    v1=load_v1();variants=[('v1','b4-v1',v1,False),('new-semantics-old-path',a.candidate,v1,False),
        ('semantic-only',a.candidate,lambda:Guide(temporal=False),True),('fusion-no-temporal',a.candidate,lambda:Guide(temporal=False),False),
        ('fusion-temporal',a.candidate,Guide,False)]
    summaries=[]
    for tag,model,constructor,strip in variants:
        source=json.loads((a.cache/model/(a.split+'-frames.json')).read_text(encoding='utf-8'));groups=collections.defaultdict(list)
        for r in source:groups[(r['case'],r['camera'])].append(r)
        rows=[];tiles=[]
        for (case,camera),frames in groups.items():
            g=constructor();last=None;switches=0;status=collections.Counter();directions=collections.Counter();valid=0;outside=0;confirmed=0;head_checked=0;head_violations=0;times=[]
            frames.sort(key=lambda r:r['time_s'])
            for i,r in enumerate(frames):
                p=dict(np.load(r['cache']));p['detections']=[] if strip else r['detections'];frame=cv2.imread(r['image'])
                start=time.perf_counter();result=g(frame,p,r['time_s']);ms=(time.perf_counter()-start)*1000;times.append(ms)
                shape=p['classes'].shape;points=[(round(x*shape[1]),round(y*shape[0])) for x,y in result['path']]
                check=validate_path(points,result['_free']);valid+=int(check['valid']);outside+=check['outside_free_pixels']
                status[result['status']]+=1;directions[result['direction']]+=1
                if result['direction']!='UNKNOWN':
                    confirmed+=1
                    if last and last!=result['direction']:switches+=1
                    last=result['direction']
                # Switch count alone is never an accuracy/stability success metric.
                label_path=Path(r['image']).parent.parent/'head-labels'/Path(r['image']).name
                label_check=None
                if camera=='head' and label_path.exists() and points:
                    gt=cv2.imread(str(label_path))[:,:,2];gt=cv2.resize(gt,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
                    label_check=validate_path(points,np.isin(gt,[0,3,6,17]));head_checked+=label_check['checked_pixels'];head_violations+=label_check['outside_free_pixels']
                row={'variant':tag,'model':model,'case':case,'camera':camera,'time_s':r['time_s'],'planner_ms':ms,
                     **{k:v for k,v in result.items() if not k.startswith('_')},'current_mask_validation':check,'human_label_path_check':label_check}
                rows.append(row)
                if camera=='chest' and i in (0,len(frames)//2,len(frames)-1):
                    view=draw(frame,result);dest=a.output/a.split/tag;dest.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(dest/(case+'-'+str(i)+'.jpg')),view)
                    tile=cv2.resize(view,(480,270));cv2.putText(tile,case+f' t={r["time_s"]:.2f}',(5,260),0,.5,(255,255,255),1);tiles.append(tile)
            summaries.append({'variant':tag,'model':model,'case':case,'camera':camera,'frames':len(frames),
                'status':dict(status),'directions':dict(directions),'non_unknown_direction_frames':confirmed,'direction_changes_between_confirmed':switches,
                'paths_valid_in_current_prediction':valid,'path_pixels_outside_current_prediction':outside,
                'human_label_path_pixels':head_checked,'human_label_path_violation_pixels':head_violations,
                'planner_p50_ms':float(np.median(times)),'planner_p95_ms':float(np.percentile(times,95))})
            print(tag,case,camera,dict(status),dict(directions),'switches',switches,flush=True)
        (a.output/(a.split+'-'+tag+'.jsonl')).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
        if tiles:
            while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(a.output/(a.split+'-'+tag+'-contact.jpg')),np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,len(tiles),3)]))
    (a.output/(a.split+'-summary.json')).write_text(json.dumps(summaries,indent=2),encoding='utf-8')

if __name__=='__main__':main()
