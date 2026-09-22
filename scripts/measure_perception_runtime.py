"""Cold-start research runtime memory; run alone for meaningful timing."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,psutil

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/cold-cuda.json'));a=ap.parse_args()
    process=psutil.Process();baseline=process.memory_info().rss;start=time.perf_counter()
    import torch
    from prts_core.models import Perception
    from prts_core.guidance import Guide
    torch.cuda.reset_peak_memory_stats()
    m=Perception(segmentation='mask2former-mapillary',long_side=1024,square=False,precision='fp16')
    load_ms=(time.perf_counter()-start)*1000;loaded=process.memory_info().rss;rows=[]
    for manifest in sorted(Path('outputs/stage2/perception/data').glob('dev-*/manifest.json')):
        data=json.loads(manifest.read_text())
        for index in [0,len(data['frames'])-1]:
            r=data['frames'][index];f=cv2.imread(str(manifest.parent/r['file']));start=time.perf_counter();p=m(f);g=Guide()(f,p,0)
            rows.append({'case':data['case'],'source_time_s':r['time_s'],'pipeline_ms':(time.perf_counter()-start)*1000,
                'rss':process.memory_info().rss,'cuda_allocated':torch.cuda.memory_allocated(),'cuda_reserved':torch.cuda.memory_reserved(),
                'status':g['status'],'direction':g['direction'],'timings':p['timings']})
    info=process.memory_info();result={'load_ms':load_ms,'baseline_rss':baseline,'after_load_rss':loaded,'final_rss':info.rss,
        'peak_process_working_set':getattr(info,'peak_wset',None),'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved':torch.cuda.max_memory_reserved(),'rows':rows,'platform':'Windows / RTX4070SUPER; not Apple unified memory',
        'conservative_peak_rss_plus_reserved_bytes':getattr(info,'peak_wset',info.rss)+torch.cuda.max_memory_reserved(),
        'limitations':'RSS includes Python and libraries; CUDA counters omit driver/context overhead. Separate GPU/RAM sums are screening figures, not Apple footprint.'}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
