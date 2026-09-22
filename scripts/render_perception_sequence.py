"""Run every source frame and export both source-time and measured-time videos.

Offline comparison: both models process the same contiguous 15 Hz source. The
source-time video is explicitly accelerated relative to measured processing.
The measured-time video holds each observation for the actual pipeline duration.
"""
import argparse,importlib.util,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from prts_core.models import Perception
from prts_core.guidance import Guide,draw
from prts_core.semantics import semantic_view
from replay_perception_comparison import load_v1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=Path('outputs/stage2/perception/data/dev-turn/manifest.json'))
    ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/continuous-turn'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text());assert manifest['frame_stride']==1
    frames=manifest['frames'];numbers=[int(Path(r['file']).stem) for r in frames]
    assert numbers==list(range(numbers[0],numbers[0]+len(numbers)))
    old=Perception(segmentation='segformer-sidewalk',long_side=512,square=True)
    new=Perception(segmentation='mask2former-mapillary',long_side=1024,square=False,precision='fp16')
    old_guide=load_v1()();new_guide=Guide();rows=[]
    fourcc=cv2.VideoWriter_fourcc(*'mp4v');source=cv2.VideoWriter(str(a.output/'source-time.raw.mp4'),fourcc,15,(1280,768))
    measured=cv2.VideoWriter(str(a.output/'processing-time.raw.mp4'),fourcc,15,(1280,768));duration=0.;written=0
    for i,r in enumerate(frames):
        start=time.perf_counter();frame=cv2.imread(str(a.manifest.parent/r['file']))
        po=old(frame);go=old_guide(frame,po,r['time_s']);old_ms=(time.perf_counter()-start)*1000
        next_start=time.perf_counter();pn=new(frame);gn=new_guide(frame,pn,r['time_s']);new_ms=(time.perf_counter()-next_start)*1000
        panels=[cv2.resize(frame,(640,360)),cv2.resize(semantic_view(frame,pn),(640,360)),
                cv2.resize(draw(frame,go),(640,360)),cv2.resize(draw(frame,gn),(640,360))]
        for panel,label in zip(panels,['SOURCE: chest camera','NEW: dense semantic regions',
                f'V1 B4 | {go["status"]} / {go["direction"]}',f'NEW M2 | {gn["status"]} / {gn["direction"]}']):
            cv2.rectangle(panel,(0,0),(640,22),(10,12,15),-1);cv2.putText(panel,label,(6,16),0,.43,(255,255,255),1)
        mosaic=np.concatenate([np.concatenate(panels[:2],1),np.concatenate(panels[2:],1)])
        elapsed=time.perf_counter()-start;duration+=elapsed
        for mode,writer in [('SOURCE TIME - offline accelerated processing',source),('MEASURED PROCESSING TIME - source slowed',measured)]:
            view=np.concatenate([mosaic,np.zeros((48,1280,3),np.uint8)])
            cv2.putText(view,f'{mode} | source {r["time_s"]:.2f}s / frame {i} | processing {duration:.2f}s',(8,740),0,.5,(240,240,240),1)
            cv2.putText(view,f'V1 {old_ms:.0f}ms | NEW {new_ms:.0f}ms | image-space candidates; no metric clearance | SANPO CC-BY-4.0',(8,761),0,.48,(180,200,220),1)
            if mode.startswith('SOURCE'):writer.write(view)
            else:
                count=max(1,round(duration*15)-written)
                for _ in range(count):writer.write(view)
                written+=count
            if i in (0,30,60,90,120,150,len(frames)-1) and mode.startswith('SOURCE'):cv2.imwrite(str(a.output/f'frame-{i:06}.jpg'),view)
        row={'source_frame':i,'source_time_s':r['time_s'],'source_sha256':r['image_sha256'],'v1_ms':old_ms,'new_ms':new_ms,
             'pipeline_ms':elapsed*1000,'processing_elapsed_s':duration,
             'v1':{k:v for k,v in go.items() if not k.startswith('_')},'new':{k:v for k,v in gn.items() if not k.startswith('_')}}
        rows.append(row)
        if i%15==0:print(i,gn['status'],gn['direction'],round(new_ms),flush=True)
    source.release();measured.release()
    (a.output/'frames.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
    (a.output/'manifest.json').write_text(json.dumps({'source_manifest':str(a.manifest),'source_fps':15,'contiguous_frames':len(rows),
        'source_duration_s':len(rows)/15,'measured_processing_s':duration,'processing_video_duration_s':written/15,
        'source_video_processing_acceleration':duration/(len(rows)/15),'mode':'offline same-input every-frame comparison, no drops',
        'timing_scope':'image decode, both perception+guidance passes and mosaic; file encoding excluded',
        'new_processing_p50_ms':float(np.median([r['new_ms'] for r in rows])),
        'new_processing_p95_ms':float(np.percentile([r['new_ms'] for r in rows],95))},indent=2))
    for name in ['source-time','processing-time']:
        subprocess.run(['D:/ffmpeg/bin/ffmpeg.exe','-y','-v','error','-i',str(a.output/(name+'.raw.mp4')),
            '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/(name+'.mp4'))],check=True)

if __name__=='__main__':main()
