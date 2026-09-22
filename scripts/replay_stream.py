"""Run real local models on timestamped recorded camera/PCM inputs at original speed.

Scenario JSON: source, start/end (original video seconds), optional roi and
ignore_regions, user_audio [{file, at}], text_commands [{text, at}], navigate.
Audio is delivered as 100 ms chunks. Source crops and artificial command timing
are recorded in the manifest; this is replay, not an iPhone or microphone capture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import psutil
import soundfile as sf
from prts_core.interfaces import VideoFrame,AudioChunk
from prts_core.local_backend import LocalBackend
from prts_core.streaming import StreamingCore
from prts_core.speech_stream import SpeechStream


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--segmentation',default='yolo26s-sem.pt')
    ap.add_argument('--asr',default='medium');ap.add_argument('--tts',action='store_true')
    ap.add_argument('--fps',type=float,default=4);ap.add_argument('--long-side',type=int,default=768)
    ap.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
    ap.add_argument('--native-library',type=Path)
    ap.add_argument('--native-server',type=Path)
    ap.add_argument('--projector',default='mmproj-model-f16.gguf')
    ap.add_argument('--waiting-mode',choices=['brain','legacy'],default='brain')
    ap.add_argument('--brain-strategy',choices=['direct','scene_agent'],default='direct')
    ap.add_argument('--sampling',choices=['greedy','recommended'],default='recommended')
    ap.add_argument('--native-threads',type=int,default=4)
    ap.add_argument('--language-file')
    ap.add_argument('--model-subdir',default='minicpm-v-4.6-gguf')
    ap.add_argument('--onnx-semantic-dir',type=Path);ap.add_argument('--onnx-detector-dir',type=Path)
    ap.add_argument('--onnx-provider',default='CPUExecutionProvider')
    ap.add_argument('--predecode',action='store_true',help='Cache lossless source frames before the timed replay, avoiding video seek overhead')
    ap.add_argument('--input-cache',type=Path,help='Reuse a previously verified input-frames manifest')
    ap.add_argument('--memory-library',type=Path,help='Optional Windows DXGI memory measurement DLL')
    ap.add_argument('--ocr-profile',choices=['bounded_upright','bounded_upright_no_arena'],default='bounded_upright_no_arena')
    args=ap.parse_args()
    if (args.output/'summary.json').exists():
        raise FileExistsError('Use a new output directory to preserve this replay.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'run-arguments.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8')
    snapshot=args.output/'implementation';snapshot.mkdir(exist_ok=True)
    hashes={}
    for source_file in sorted((ROOT/'prts_core').glob('*.py')):
        data=source_file.read_bytes();(snapshot/source_file.name).write_bytes(data)
        hashes[source_file.name]=hashlib.sha256(data).hexdigest()
    (args.output/'implementation-sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (args.output/'runner.py.snapshot').write_bytes(Path(__file__).read_bytes())
    if args.native_library:
        with args.native_library.open('rb') as file:native_sha=hashlib.file_digest(file,'sha256').hexdigest()
        (args.output/'native-library.json').write_text(json.dumps(dict(path=str(args.native_library),sha256=native_sha),indent=2))
    native_log=(args.output/'native-stderr.log').open('w',encoding='utf-8')
    os.dup2(native_log.fileno(),2)
    case=json.loads(args.scenario.read_text(encoding='utf-8'))
    source=Path(case['source']);source=source if source.is_absolute() else ROOT/source
    cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    if not cap.isOpened():raise RuntimeError('Cannot open scenario video')
    original_fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/original_fps
    begin=case.get('start',0);end=min(duration,case.get('end',duration));duration=end-begin
    def prepare_image(frame):
        if case.get('roi'):
            h,w=frame.shape[:2];x1,y1,x2,y2=case['roi'];frame=frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)].copy()
        for x1,y1,x2,y2 in case.get('ignore_regions',[]):
            h,w=frame.shape[:2];frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]=0
        return frame
    cached=[]
    if args.input_cache:
        cached=json.loads((args.input_cache/'manifest.json').read_text(encoding='utf-8'))
        if len(cached)!=int(np.ceil(duration*args.fps)):raise ValueError('Cache duration or FPS differs')
        for i,entry in enumerate(cached):
            if abs(entry['source_s']-(begin+i/args.fps))>.001:raise ValueError('Cache source clock differs')
            if hashlib.sha256(Path(entry['file']).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Cache hash differs')
    elif args.predecode:
        cache=args.output/'input-frames';cache.mkdir(parents=True,exist_ok=True)
        for i in range(int(np.ceil(duration*args.fps))):
            stamp=begin+i/args.fps;cap.set(cv2.CAP_PROP_POS_MSEC,stamp*1000);ok,frame=cap.read()
            if not ok:raise RuntimeError('Cannot predecode scenario frame')
            file=cache/f'{i:06}.png';cv2.imwrite(str(file),prepare_image(frame),[cv2.IMWRITE_PNG_COMPRESSION,1])
            cached.append(dict(file=str(file),at_s=i/args.fps,source_s=stamp,
                               sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        (cache/'manifest.json').write_text(json.dumps(cached,indent=2),encoding='utf-8')
        print(json.dumps(dict(predecoded_frames=len(cached),inference_not_started=True)),flush=True)
    from prts_core.resource_monitor import ResourceMonitor
    monitor=ResourceMonitor(args.output/'memory.jsonl',args.memory_library)
    backend=LocalBackend(ROOT/'models',args.device,args.segmentation,args.long_side,args.asr,prewarm_audio=True,
                         native_library=args.native_library,precision=args.precision,native_projector=args.projector,
                         onnx_semantic_dir=args.onnx_semantic_dir,onnx_detector_dir=args.onnx_detector_dir,
                         onnx_provider=args.onnx_provider,waiting_mode=args.waiting_mode,
                         native_language_file=args.language_file,native_model_subdir=args.model_subdir,
                         native_server=args.native_server,native_server_log=args.output/'server.log',
                         native_sampling=args.sampling,brain_strategy=args.brain_strategy,native_threads=args.native_threads,ocr_profile=args.ocr_profile)
    # Prewarm vision before the recorded stream starts; account separately.
    if cached:first=cv2.imread(cached[0]['file']);ok=first is not None;cap.release()
    else:cap.set(cv2.CAP_PROP_POS_MSEC,begin*1000);ok,first=cap.read()
    if not ok:raise RuntimeError('Cannot read starting frame')
    prep=time.perf_counter();backend.perception(first);warmup_s=time.perf_counter()-prep
    if args.native_library or args.native_server:backend._load_brain()
    speech=SpeechStream(args.output/'speech') if args.tts else None
    interesting=[]
    def event(e):
        if speech:speech.accept(e)
        if e['type'] in ('answer','target_observed','transcript','request_error','worker_error','brain_observation'):
            interesting.append(e)
            display={k:v for k,v in e.items() if k not in ('input','brain_decision')}
            print(json.dumps(display,ensure_ascii=False),flush=True)
    core=StreamingCore(backend,event,trace_path=args.output/'events.jsonl',max_frame_age_s=2).start()
    audio=[]
    for entry in case.get('user_audio',[]):
        p=Path(entry['file']);p=p if p.is_absolute() else ROOT/p
        samples,sr=sf.read(p,dtype='float32',always_2d=True)
        if sr!=16000:raise ValueError('Replay WAV must be 16 kHz')
        mono=samples.mean(1)
        for offset in range(0,len(mono),1600):
            audio.append((entry['at']+offset/16000,mono[offset:offset+1600],
                          offset+1600>=len(mono),entry.get('role','user')))
    audio.sort(key=lambda a:a[0]);commands=sorted(case.get('text_commands',[]),key=lambda x:x['at'])
    start=time.monotonic();next_frame=0.;audio_seq=0;frame_seq=0;peak_rss=0;peak_process_tree_rss=0
    writer=None;timeline=[];saved=set();last_recorded=None
    if case.get('navigate'):core.push_text('开始导航')
    try:
        while time.monotonic()-start<duration:
            elapsed=time.monotonic()-start
            while audio and audio[0][0]+len(audio[0][1])/16000<=elapsed:
                at,samples,final,role=audio.pop(0)
                core.push_audio(AudioChunk(start+at,audio_seq,samples,role=role,end_utterance=final));audio_seq+=1
            while commands and commands[0]['at']<=elapsed:
                cmd=commands.pop(0);core.push_text(cmd['text'],cmd.get('role','user'))
            if elapsed>=next_frame:
                frame_time=elapsed
                if cached:
                    entry=cached[min(len(cached)-1,int(elapsed*args.fps))]
                    frame=cv2.imread(entry['file']);frame_time=entry['at_s']
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC,(begin+elapsed)*1000);ok,frame=cap.read()
                    if not ok:break
                    frame=prepare_image(frame)
                frame_seq+=1;core.push_video(VideoFrame(start+frame_time,frame_seq,frame))
                processed=backend.last_view
                view=processed[2].copy() if processed else frame.copy()
                width=960;height=round(view.shape[0]*width/view.shape[1]/2)*2
                view=cv2.resize(view,(width,height))
                age=elapsed-(processed[1]-start) if processed else 0
                view=cv2.copyMakeBorder(view,62,0,0,0,cv2.BORDER_CONSTANT,value=(20,22,28));height+=62
                cv2.putText(view,f'PRTS Core | recorded input 1x | source {begin+elapsed:.2f}s',(12,24),0,.60,(240,240,240),1)
                cv2.putText(view,f'wall {elapsed:.2f}s | result age {age:.2f}s | frames {core.stats["frames_processed"]}',(12,50),0,.55,(240,240,240),1)
                if writer is None:
                    writer=cv2.VideoWriter(str(args.output/'annotated_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(width,height))
                # Fill actual wall-clock gaps with the last displayed result.
                count=max(1,round(elapsed*args.fps)-len(timeline))
                for _ in range(count):
                    writer.write(last_recorded if last_recorded is not None else view)
                    timeline.append(dict(frame=len(timeline),wall_s=len(timeline)/args.fps,
                                         result_sequence=processed[0] if processed else None))
                last_recorded=view
                if int(elapsed)//5 not in saved:
                    saved.add(int(elapsed)//5);cv2.imwrite(str(args.output/f'screenshot_{int(elapsed):03}.jpg'),view)
                next_frame=(int(elapsed*args.fps)+1)/args.fps
                peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
                resident=psutil.Process().memory_info().rss
                for child in psutil.Process().children(recursive=True):
                    try:resident+=child.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak_process_tree_rss=max(peak_process_tree_rss,resident)
            time.sleep(.005)
    finally:
        cap.release()
        if writer:writer.release()
        core.close(90)
        if speech:speech.close()
        if backend._brain is not None and hasattr(backend._brain,'close'):backend._brain.close()
        memory_report=monitor.close()
    if writer:
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(args.output/'annotated_raw.mp4'),'-c:v','libx264',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'annotated.mp4')],check=True)
    report=dict(scenario=case,recorded_input=True,live_device=False,input_speed=1,started_monotonic_s=start,
                source_start_s=begin,source_end_s=end,wall_s=time.monotonic()-start,warmup_s=warmup_s,
                stats=core.stats,peak_sampled_rss_bytes=peak_rss,interesting_events=interesting,
                peak_sampled_process_tree_rss_bytes=peak_process_tree_rss,
                memory_scope='Sampled Python plus child-server RSS; excludes DXGI allocations and is not total Apple footprint.',
                video_audio='video currently silent; live-generated speech files have independent timing metadata')
    import onnxruntime
    report['runtime']=dict(onnxruntime=onnxruntime.__version__,available_providers=onnxruntime.get_available_providers(),
                           torch_imported='torch' in sys.modules,decoder_threads=1)
    report['predecoded_camera_frames']=bool(cached)
    report['memory']=memory_report
    report['brain_strategy']=args.brain_strategy;report['sampling']=args.sampling
    report['native_threads']=args.native_threads
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(stats=core.stats,peak_sampled_rss_bytes=peak_rss,output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':main()
