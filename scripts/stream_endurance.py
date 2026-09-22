"""Measure the complete resident research core under repeated recorded inputs.

This measures scheduling, memory and stability. Looped recordings and injected
commands are explicitly not an independent field-safety or recognition benchmark.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
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
    ap.add_argument('--duration',type=float,default=1200)
    ap.add_argument('--device',choices=['cpu','cuda','dml'],default='cpu')
    ap.add_argument('--native-library',type=Path,default=ROOT/'outputs/prts-native-build/libprts_vlm.dll')
    ap.add_argument('--ocr-profile',choices=['bounded_upright','bounded_upright_no_arena'],default='bounded_upright')
    ap.add_argument('--projector',default='mmproj-model-f16.gguf')
    ap.add_argument('--include-queue',action='store_true',help='Exercise real queue-photo OCR alongside navigation, speech and vehicle waits')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--semantic-dir',type=Path,default=ROOT/'outputs/stage2/perception/onnx/mask2former-1024')
    args=ap.parse_args()
    if (args.output/'summary.json').exists():raise FileExistsError('Use a new output directory to preserve the previous run.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'run-arguments.json').write_text(json.dumps(vars(args),default=str,indent=2),encoding='utf-8')
    (args.output/'runner.py.snapshot').write_bytes(Path(__file__).read_bytes())
    snapshot=args.output/'implementation';snapshot.mkdir(exist_ok=True)
    hashes={}
    for source in sorted((ROOT/'prts_core').glob('*.py')):
        data=source.read_bytes();(snapshot/source.name).write_bytes(data)
        hashes[source.name]=hashlib.sha256(data).hexdigest()
    (args.output/'implementation-sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    native_log=(args.output/'native-stderr.log').open('w',encoding='utf-8');os.dup2(native_log.fileno(),2)
    torch=None
    dxgi=None
    if args.device=='dml':
        from prts_core.windows_memory import WindowsGPUMemory
        dxgi=WindowsGPUMemory(ROOT/'outputs/native-toolchain/prts_gpu_memory.dll')
        dxgi.sample()
    if args.device=='cuda':
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('CUDA research profile requested but unavailable')
        torch.cuda.reset_peak_memory_stats()
    memory=[];done=threading.Event();process=psutil.Process();started=time.monotonic()
    def sample():
        with (args.output/'memory.jsonl').open('w',encoding='utf-8') as f:
            while not done.wait(.1):
                info=process.memory_info();row=dict(elapsed_s=time.monotonic()-started,rss=info.rss,
                    peak_working_set=getattr(info,'peak_wset',info.rss),gpu_allocated=0,gpu_reserved=0,
                    gpu_local_usage=0,gpu_nonlocal_usage=0)
                if dxgi:row.update(dxgi.sample())
                if torch:
                    row.update(gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved())
                memory.append(row);f.write(json.dumps(row)+'\n')
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    backend=core=speech=None;counts=Counter();errors=[];samples=[];caps={}
    try:
        kwargs=dict(model_dir=ROOT/'models',device='cpu' if args.device=='dml' else args.device,asr_variant='sensevoice',prewarm_audio=True,
                    native_library=args.native_library,ocr_profile=args.ocr_profile,native_projector=args.projector)
        if args.device in ('cpu','dml'):
            kwargs.update(onnx_semantic_dir=args.semantic_dir,
                          onnx_detector_dir=ROOT/'outputs/stage2/perception/onnx/yolo11n-rect',
                          onnx_provider='DmlExecutionProvider' if args.device=='dml' else 'CPUExecutionProvider')
        else:kwargs.update(segmentation='mask2former-mapillary',long_side=1024,precision='fp16')
        backend=LocalBackend(**kwargs);backend._load_brain()
        bus=json.loads((ROOT/'outputs/stage2/streaming/bus-case.json').read_text(encoding='utf-8'))
        notice=json.loads((ROOT/'outputs/stage2/streaming/notice-case.json').read_text(encoding='utf-8'))
        cases=[notice,bus]
        for case in cases:
            caps[case['source']]=cv2.VideoCapture(case['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        speech=SpeechStream(args.output/'speech')
        def event(e):
            counts[e['type']]+=1;speech.accept(e)
            if e['type'] in ('worker_error','request_error'):errors.append(e)
            if e['type'] in ('answer','target_observed','target_candidate','transcript') and len(samples)<40:samples.append(e)
        core=StreamingCore(backend,event,trace_path=args.output/'events.jsonl').start()
        media_started=time.monotonic();frame_seq=0;audio_seq=0;last_frame=-1.;last_cycle=-1;audio=[];next_report=30
        queue_started_cycle=-1;queue_image=None
        queue_cases=[('outputs/stage2/queue-holdout/data/commons-bank-queue.jpg','C002'),
                     ('outputs/stage2/queue-holdout/data/commons-hk-foodcourt.jpg','215'),
                     ('outputs/stage2/queue-holdout/data-v2/masmitra.jpg','B003')]
        while time.monotonic()-media_started<args.duration:
            elapsed=time.monotonic()-media_started;cycle=int(elapsed//60);phase=elapsed%60
            if cycle!=last_cycle:
                core.push_text('开始导航');last_cycle=cycle
                queue_image=None
                # Three real user utterances, delivered as continuous 100 ms PCM chunks.
                for clip,at in [('notice_user.wav',3),('bus_user.wav',24),('road_user.wav',46)]:
                    wave,rate=sf.read(ROOT/'outputs/cases'/clip,dtype='float32',always_2d=True)
                    if rate!=16000:raise ValueError('Prepared audio must be 16 kHz')
                    wave=wave.mean(1)
                    for i in range(0,len(wave),1600):audio.append((cycle*60+at+i/16000,wave[i:i+1600],i+1600>=len(wave)))
            while audio and audio[0][0]+len(audio[0][1])/16000<=elapsed:
                at,wave,final=audio.pop(0);core.push_audio(AudioChunk(media_started+at,audio_seq,wave,role='user',end_utterance=final));audio_seq+=1
            if args.include_queue and phase>=45 and queue_started_cycle!=cycle:
                file,target=queue_cases[cycle%len(queue_cases)];queue_image=cv2.imread(str(ROOT/file))
                if queue_image is None:raise RuntimeError('Queue source unavailable')
                core.push_text(f'等{target}号，叫到时提醒我');queue_started_cycle=cycle
            if elapsed-last_frame>=.5:
                case=notice if phase<22 or phase>=45 else bus
                length=case['end']-case['start'];source_s=case['start']+(phase if case is notice else phase-22)%length
                cap=caps[case['source']];cap.set(cv2.CAP_PROP_POS_MSEC,source_s*1000);ok,bgr=cap.read()
                if not ok:raise RuntimeError('Replay frame unavailable')
                if case.get('roi'):
                    h,w=bgr.shape[:2];x1,y1,x2,y2=case['roi'];bgr=bgr[int(y1*h):int(y2*h),int(x1*w):int(x2*w)].copy()
                for x1,y1,x2,y2 in case.get('ignore_regions',[]):
                    h,w=bgr.shape[:2];bgr[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]=0
                if args.include_queue and phase>=45:bgr=queue_image.copy()
                core.push_video(VideoFrame(media_started+elapsed,frame_seq,bgr));frame_seq+=1;last_frame=elapsed
            if elapsed>=next_report:
                print(json.dumps(dict(elapsed_s=round(elapsed),rss_gb=round(process.memory_info().rss/1e9,3),events=dict(counts))),flush=True)
                next_report+=30
            time.sleep(.005)
        core.push_text('停止导航');core.push_text('取消等待');core.close(90);core=None
        speech.close();speech=None
    finally:
        if core:core.close(90)
        if speech:speech.close()
        if backend and backend._brain and hasattr(backend._brain,'close'):backend._brain.close()
        for cap in caps.values():cap.release()
        done.set();monitor.join(5)
    peaks=dict(rss=max(r['rss'] for r in memory),working_set=max(r['peak_working_set'] for r in memory),
               gpu_allocated=max(r['gpu_allocated'] for r in memory),gpu_reserved=max(r['gpu_reserved'] for r in memory),
               gpu_local_usage=max(r['gpu_local_usage'] for r in memory),gpu_nonlocal_usage=max(r['gpu_nonlocal_usage'] for r in memory))
    envelope=max(max(r['rss']+r['gpu_reserved']+r['gpu_local_usage']+r['gpu_nonlocal_usage'] for r in memory),peaks['working_set'])
    report=dict(profile=args.device,duration_s=args.duration,total_wall_s=time.monotonic()-started,
                peaks=peaks,simultaneous_rss_plus_gpu_reserved=envelope,
                with_assumed_frontend_2gb=envelope+2_000_000_000,events=dict(counts),errors=errors,examples=samples,
                gpu_measurement='DXGI current process adapter 0 local + nonlocal usage; matches DML device_id=0 (nonlocal may overlap RSS)' if dxgi else 'torch counters' if torch else 'CPU only',
                semantic_dir=str(args.semantic_dir),
                native_language=backend.native_language,ocr_profile=args.ocr_profile,
                native_projector=args.projector,
                projector_sha256=hashlib.sha256((ROOT/'models/minicpm-v-4.6-gguf'/args.projector).read_bytes()).hexdigest(),
                native_library=str(args.native_library),native_library_sha256=hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
                implementation_sha256=hashes,
                torch_imported='torch' in sys.modules,
                decoder_threads=1,
                includes_queue=args.include_queue,
                queue_sources=queue_cases if args.include_queue else [],
                provenance='Repeated prerecorded dissertation videos, real extracted user speech, injected start/stop/cancel commands; no physical microphone/camera/speaker.',
                budget_note='Dedicated GPU memory plus RSS is a conservative research envelope, not measured Apple unified footprint. Frontend 2 GB is a reservation, not an observed application.')
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(peaks=peaks,envelope=envelope,errors=len(errors),output=str(args.output))))


if __name__=='__main__':main()
