"""Cache actual same-input semantics, detections, timings and human-label metrics.

Chest-camera imagery is unlabelled; head-camera labels are evaluated separately.
This never interprets planner availability or larger green area as accuracy.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch,psutil
from prts_core.models import Perception
from prts_core.semantics import semantic_view
from perception_metrics import label_metrics

CONFIGS={
 'b4-v1':('segformer-sidewalk',512,True),
 'b4-aspect':('segformer-sidewalk',1024,False),
 'yolo26s':('yolo26s-sem.pt',1024,False),
 'yolo26m':('yolo26m-sem.pt',1024,False),
 'mask2former':('mask2former-mapillary',1024,False),
 'mask2former-fp16':('mask2former-mapillary',1024,False),
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',choices=CONFIGS,default=list(CONFIGS))
    ap.add_argument('--data',type=Path,default=Path('outputs/stage2/perception/data'))
    ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/comparison'))
    ap.add_argument('--split',choices=['development','heldout'],default='development')
    ap.add_argument('--stride',type=int,default=1);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for path in sorted(a.data.glob('*/manifest.json')):
        item=json.loads(path.read_text())
        if item['split']==a.split:manifests.append((path,item))
    if not manifests:raise ValueError('no manifests for split')
    for tag in a.models:
        name,size,square=CONFIGS[tag];folder=a.output/tag;folder.mkdir(parents=True,exist_ok=True)
        m=Perception(segmentation=name,long_side=size,square=square,precision='fp16' if tag.endswith('-fp16') else 'fp32')
        torch.cuda.reset_peak_memory_stats();rows=[];rss=0;tiles=[]
        for path,manifest in manifests:
            case=manifest['case'];dest=folder/case;dest.mkdir(exist_ok=True)
            chest=manifest['frames'][::4] if manifest['frame_stride']==1 else manifest['frames']
            sources=[('chest',r) for r in chest[::a.stride]]+[('head',r) for r in manifest['human_label_frames']]
            for index,(camera,r) in enumerate(sources):
                stem=camera+'-'+Path(r['file']).stem;cache=dest/(stem+'.npz');meta=dest/(stem+'.json')
                if cache.exists() and meta.exists():rows.append(json.loads(meta.read_text()));continue
                frame=cv2.imread(str(path.parent/r['file']));start=time.perf_counter();p=m(frame);elapsed=(time.perf_counter()-start)*1000
                row={'case':case,'split':a.split,'camera':camera,'time_s':r['time_s'],'image':str(path.parent/r['file']),
                     'source_image_sha256':r['image_sha256'],'cache':str(cache),'model':tag,'model_config':[name,size,square],
                     'frame_ms':elapsed,**p['timings'],'detections':p['detections'],'labels':p['labels'],'regions':p['regions']}
                if camera=='head':row['label_metrics']=label_metrics(p,path.parent/r['label']);row['annotation_type']=r['annotation_type']
                np.savez_compressed(cache,**{k:p[k] for k in ['classes','semantic_groups','sidewalk_class','sidewalk','road','confidence']})
                meta.write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8');rows.append(row)
                if index in (0,len(chest)//2,len(chest)-1):
                    view=semantic_view(frame,p);cv2.imwrite(str(dest/(stem+'.jpg')),view)
                    tile=cv2.resize(view,(480,270));cv2.putText(tile,case+' '+str(round(r['time_s'],2)),(8,20),0,.52,(255,255,255),1);tiles.append(tile)
                rss=max(rss,psutil.Process().memory_info().rss)
            print(tag,case,len(sources),'done',flush=True)
        summary={'model':tag,'split':a.split,'configuration':[name,size,square],'frames':len(rows),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
            'peak_process_rss_bytes':rss,'timing_frame_p50_ms':float(np.median([r['frame_ms'] for r in rows])),
            'timing_frame_p95_ms':float(np.percentile([r['frame_ms'] for r in rows],95))}
        labelled=[r['label_metrics'] for r in rows if 'label_metrics' in r]
        if labelled:
            totals={k:sum(r[k] for r in labelled) for k in ['tp','fp','fn','roadway_admitted_pixels','fixed_admitted_pixels','dynamic_admitted_pixels']}
            summary['human_head']={**totals,'count':len(labelled),'aggregate_walkable_iou':totals['tp']/max(1,totals['tp']+totals['fp']+totals['fn'])}
        previous=folder/(a.split+'-summary.json')
        if previous.exists():
            old=json.loads(previous.read_text())
            for key in ['peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_process_rss_bytes']:summary[key]=max(summary[key],old.get(key,0))
        summary['memory_scope']='process peaks across resumed comparison runs; CPU RSS includes framework imports and prior model allocator caches; use isolated runtime measurement for budget'
        previous.write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (folder/(a.split+'-frames.json')).write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
        if tiles:
            while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(folder/(a.split+'-contact.jpg')),np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,len(tiles),3)]))
        print(json.dumps(summary),flush=True);del m;torch.cuda.empty_cache()

if __name__=='__main__':main()
