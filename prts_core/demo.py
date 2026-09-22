"""Offline backend demo: python -m prts_core.demo --scenario scenarios/example.json"""
import argparse
import json
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import psutil

from .models import ROOT,ASR,Brain,OCR,Perception
from .guidance import Guide,draw
from .tasks import Tasks
from .intent import route, read_text_request, ordered_ocr


def resolve(value,base=ROOT):
    p=Path(value)
    return p if p.is_absolute() else base/p


class Engine:
    def __init__(self,output,model_dir=ROOT/'models',device='cuda',segmentation='segformer-sidewalk',navigation=False,tts=False):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.log=(self.output/'events.jsonl').open('w',encoding='utf-8')
        self.model_dir,self.device=model_dir,device
        self.perception=Perception(model_dir,device,segmentation=segmentation)
        self.guide=Guide();self.tasks=Tasks(navigation)
        self._brain=None;self._asr=None;self._ocr=None
        self.last_guidance=None;self.last_guidance_s=-100;self.peak_rss=0
        self.started=time.perf_counter();self.records=[];self.commands=[]
        self.scene_memory=deque(maxlen=12)
        self.tts=tts;self.speech_events=[]

    @property
    def brain(self):
        if self._brain is None:self._brain=Brain(self.model_dir,self.device)
        return self._brain

    @property
    def asr(self):
        if self._asr is None:self._asr=ASR(self.model_dir)
        return self._asr

    @property
    def ocr(self):
        if self._ocr is None:self._ocr=OCR()
        return self._ocr

    def emit(self,event):
        event.setdefault('elapsed_s',time.perf_counter()-self.started)
        self.log.write(json.dumps(event,ensure_ascii=False)+'\n');self.log.flush()
        if self.tts and event['type'] in ('answer','target_observed','navigation_started','navigation_stopped','wait_started','wait_cancelled','need_target','unsupported_task','guidance'):
            self.speech_events.append(dict(event))
        if event['type']!='observation':
            concise={k:v for k,v in event.items() if k in ('type','observation_s','text','question','action','kind','target','status','direction','source')}
            print(json.dumps(concise,ensure_ascii=False),flush=True)
        return event

    def command(self,command,frame,now,frame_provider=None):
        started=time.perf_counter()
        evidence={}
        if 'audio' in command:
            evidence=self.asr(resolve(command['audio']))
            text=evidence['text'];self.emit(dict(type='transcript',observation_s=now,role='user',**evidence))
        else:text=command['text']
        if not text.strip():
            self.emit(dict(type='no_speech',observation_s=now,text='没有识别到用户指令'));return
        intent=route(text,self.tasks.wait)
        if intent is None:
            intent=self.brain.intent(text);intent['source']='local_language_model'
        self.emit(dict(type='intent',observation_s=now,utterance=text,**intent))
        kind,message=self.tasks.command(intent,now)
        self.emit(dict(type=kind,observation_s=now,text=message))
        record=dict(time_s=now,utterance=text,intent=intent,asr=evidence)
        if intent['action']=='ask':
            if frame_provider:
                fresh=frame_provider()
                if fresh:now,frame=fresh
                else:
                    self.emit(dict(type='answer',question=text,observation_s=now,text='当前没有新鲜画面，请调整相机后再问',source='stale_frame_guard'))
                    record['total_processing_ms']=(time.perf_counter()-started)*1000;self.commands.append(record);return
            ocr=self.ocr(frame)
            lines=ordered_ocr(ocr)
            if read_text_request(text) and lines:
                answer=dict(text='画面可辨认的文字：'+ '；'.join(lines)+'。',source='ocr_transcription',processing_ms=0)
            elif re.search('挡|障碍|路况|前面|眼前|周围',text):
                # Ground an open visual answer in this exact frame, not old boxes or OCR.
                current=self.perception(frame);ground=current['sidewalk_class']&(current['sidewalk']>=.5)
                h,w=ground.shape;objects=[]
                names={'person':'行人','bicycle':'自行车','car':'汽车','motorcycle':'摩托车',
                       'bus':'公交车','truck':'货车','bench':'长椅','chair':'椅子','dog':'动物','horse':'动物'}
                for d in current['detections']:
                    x1,y1,x2,y2=d['box'];cx=(x1+x2)/2
                    if d['score']<.4 or y2<.5 or not .2<cx<.8:continue
                    a=max(0,int((x1-.025)*w));b=min(w,int((x2+.025)*w))
                    c=max(0,int((y2-.025)*h));e=min(h,max(c+1,int((y2+.055)*h)))
                    near_ground=bool(ground[c:e,a:b].any())
                    if not near_ground:continue
                    objects.append(dict(label=names.get(d['label'],d['label']),position='中部' if .35<cx<.65 else '左侧' if cx<=.35 else '右侧',
                                        score=d['score'],box=d['box']))
                prompt=('用户问：'+text+'\n请结合当前画面和检测结果回答。先描述前方中部的人或物体，再补充左右两侧。'
                        '不要识别或抄写标牌文字，不推断物体一定完全挡住路，不承诺可以通行。用两三句中文。'
                        '\n当前帧人行道候选区域附近的检测：'+json.dumps(objects,ensure_ascii=False))
                answer=self.brain.generate(prompt,frame,max_tokens=160,raw_prompt=True)
                answer.update(source='perception_and_local_vlm',current_frame_objects=objects)
            else:
                context={'当前OCR文字':lines,'近期观察（可能过时）':list(self.scene_memory)[-4:],
                         '最近用户问题':[c['utterance'] for c in self.commands[-3:]],'当前等待任务':self.tasks.wait}
                answer=self.brain.generate(text,frame,context=json.dumps(context,ensure_ascii=False))
                answer['source']='local_vision_language_model'
            image_name=f'question_{len(self.commands):03}.jpg'
            cv2.imwrite(str(self.output/image_name),frame)
            record.update(answer=answer,ocr=ocr,evidence_image=image_name,evidence_time_s=now)
            self.emit(dict(type='answer',observation_s=now,question=text,evidence_image=image_name,ocr=ocr,**answer))
        record['total_processing_ms']=(time.perf_counter()-started)*1000
        self.commands.append(record)

    def process(self,frame,now,ambient=None,capture_monotonic=None):
        start=time.perf_counter();perception=self.perception(frame)
        guide=self.guide(frame,perception,now)
        if capture_monotonic is not None and time.monotonic()-capture_monotonic>1.5:
            guide.update(status='WAIT',direction='UNKNOWN',reason='stale_frame',path=[],
                         text='画面处理延迟，暂停引导并重新观察')
        record={k:v for k,v in guide.items() if not k.startswith('_')}
        record.update(time_s=now,**perception['timings'])
        if self.tasks.navigating:
            state=(guide['status'],guide['direction'])
            if state!=self.last_guidance or now-self.last_guidance_s>=4:
                self.emit(dict(type='guidance',observation_s=now,**record))
                self.last_guidance=state;self.last_guidance_s=now
        if self.tasks.wait and not self.tasks.wait['notified']:
            ocr=self.ocr(frame)
            h,w=frame.shape[:2]
            for entry in ocr:
                center=np.mean(entry['box'],axis=0)/[w,h]
                for d in perception['detections']:
                    x1,y1,x2,y2=d['box']
                    if x1<=center[0]<=x2 and y1<=center[1]<=y2:entry['vehicle_label']=d['label']
            # Dedicated vehicle crop retains fine route digits, still backed by real detections.
            if self.tasks.wait['kind'] in ('bus','train'):
                for d in perception['detections']:
                    if d['label'] not in ('bus','train'):continue
                    x1,y1,x2,y2=d['box'];crop=frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]
                    if not crop.size:continue
                    for entry in self.ocr(crop):
                        entry['vehicle_label']=d['label'];entry['crop_box']=d['box'];ocr.append(entry)
            audio_text='';audio_start=None
            if ambient:
                a=self.asr(resolve(ambient['audio']),role='ambient');audio_text=a['text'];audio_start=ambient.get('start',now)
                self.emit(dict(type='transcript',role='ambient',observation_s=now,**a))
            record['ocr']=ocr
            event=self.tasks.observe(ocr,perception['detections'],now,audio_text,audio_start)
            if event:
                image_name=f'target_{self.tasks.version:03}.jpg';cv2.imwrite(str(self.output/image_name),frame)
                event['evidence_image']=image_name
                if ambient:event['evidence_audio']=ambient['audio']
                self.emit(event)
        record['processing_ms']=(time.perf_counter()-start)*1000
        self.peak_rss=max(self.peak_rss,psutil.Process().memory_info().rss)
        self.records.append(record);self.emit(dict(type='observation',**record))
        self.scene_memory.append(dict(time_s=now,status=guide['status'],direction=guide['direction'],
                                      objects=sorted({d['label'] for d in perception['detections']})))
        return draw(frame,guide)

    def close(self):
        speech=[];tts_ms=0
        if self.tts and self.speech_events:
            from .speech import export_speech
            start=time.perf_counter();speech=export_speech(self.speech_events,self.output/'speech')
            tts_ms=(time.perf_counter()-start)*1000
            self.emit(dict(type='speech_export',files=len(speech),text='中文答复已导出为本地WAV',manifest='speech/manifest.json'))
        self.log.close()
        times=[x['processing_ms'] for x in self.records]
        summary=dict(frames=len(times),wall_s=time.perf_counter()-self.started,peak_sampled_rss_bytes=self.peak_rss,
                     processing_ms={'mean':float(np.mean(times)) if times else None,
                                    'p50':float(np.percentile(times,50)) if times else None,
                                    'p95':float(np.percentile(times,95)) if times else None},commands=self.commands,
                     states={s:sum(r['status']==s for r in self.records) for s in sorted({r['status'] for r in self.records})},
                     speech_files=len(speech),speech_export_ms=tts_ms,
                     note='Desktop measurements; media time differs from wall time. Image-space corridor, not metric navigation.')
        if self.device.startswith('cuda'):
            import torch
            summary['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            summary['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        (self.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


def file_frames(scenario):
    path=resolve(scenario['source']);start=scenario.get('start',0);end=scenario.get('end',float('inf'))
    interval=scenario.get('interval',.5);last=-float('inf')
    roi=scenario.get('roi')
    def crop(frame):
        if roi:
            h,w=frame.shape[:2];x1,y1,x2,y2=roi
            frame=frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)].copy()
        # Optional masks for a recording's app UI, specified explicitly by its manifest.
        for x1,y1,x2,y2 in scenario.get('ignore_regions',[]):
            h,w=frame.shape[:2];frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]=0
        return frame
    if path.suffix=='.json':
        manifest=json.loads(path.read_text(encoding='utf-8'))
        for entry in manifest['frames']:
            t=entry['time_s']
            if t<start or t>end or t-last<interval-1e-5:continue
            frame=cv2.imread(str(path.parent/entry['file']))
            if frame is None:raise RuntimeError(f'Cannot read {entry["file"]}')
            last=t;yield t,crop(frame)
    elif path.suffix.lower() in ('.jpg','.png','.jpeg'):
        frame=cv2.imread(str(path))
        if frame is None:raise RuntimeError(f'Cannot read {path}')
        yield start,crop(frame)
    else:
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise RuntimeError(f'Cannot open {path}')
        fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        t=start
        try:
            while t<=min(end,duration-.01):
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
                if not ok:break
                yield t,crop(frame);t+=interval
        finally:cap.release()


def replay(args):
    scenario=json.loads(Path(args.scenario).read_text(encoding='utf-8'))
    engine=Engine(args.output,args.models,args.device,args.segmentation,scenario.get('navigate',False),args.tts)
    writer=None;frames=[];commands=list(scenario.get('commands',[]));ambient=list(scenario.get('ambient',[]))
    previous_view=None;previous_time=None;video_fps=1/scenario.get('interval',.5)
    video_start=None;written_frames=0
    try:
        for now,frame in file_frames(scenario):
            while commands and commands[0].get('at',0)<=now:
                engine.command(commands.pop(0),frame,now)
            announcement=None
            if ambient and ambient[0]['at']<=now:announcement=ambient.pop(0)
            view=engine.process(frame,now,announcement)
            scale=min(1,960/max(view.shape[:2]));w=int(view.shape[1]*scale)//2*2;h=int(view.shape[0]*scale)//2*2
            view=cv2.resize(view,(w,h))
            cv2.putText(view,f'media t={now:.2f}s',(10,h-12),0,.48,(255,255,255),1)
            if writer is None:
                writer=cv2.VideoWriter(str(Path(args.output)/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                                       video_fps,(w,h))
                if not writer.isOpened():raise RuntimeError('Video output unavailable')
            if previous_view is not None:
                count=max(0,round((now-video_start)*video_fps)-written_frames)
                for _ in range(count):writer.write(previous_view)
                written_frames+=count
            else:video_start=now
            previous_view,previous_time=view,now
            if len(frames)<9 or len(engine.records)%10==0:
                frames.append(view)
                if len(frames)>12:frames.pop(6)
        if commands:engine.emit(dict(type='unprocessed_commands',count=len(commands),text='Commands lie after the last input frame'))
    finally:
        if writer:
            if previous_view is not None:writer.write(previous_view)
            writer.release()
        engine.close()
    if frames:
        chosen=[frames[i] for i in np.linspace(0,len(frames)-1,min(6,len(frames)),dtype=int)]
        cv2.imwrite(str(Path(args.output)/'preview.jpg'),np.concatenate(chosen,axis=1))


class LatestCamera:
    def __init__(self,index):
        self.cap=cv2.VideoCapture(index);self.value=None;self.running=True;self.lock=threading.Lock()
        if not self.cap.isOpened():raise RuntimeError(f'Cannot open camera {index}')
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        while self.running:
            ok,frame=self.cap.read()
            if not ok:self.running=False;break
            with self.lock:self.value=(time.monotonic(),frame)
    def latest(self):
        with self.lock:return self.value
    def close(self):
        self.running=False;self.thread.join(timeout=2);self.cap.release()


def live(args):
    camera=LatestCamera(args.camera);commands=queue.Queue();start=time.monotonic()
    engine=Engine(args.output,args.models,args.device,args.segmentation,True,args.tts)
    def input_commands():
        print('输入文字命令；输入 /mic 后录制 5 秒中文命令；/quit 结束。摄像头无需窗口。',flush=True)
        while camera.running:
            try:text=input().strip()
            except EOFError:break
            if text=='/quit':commands.put(None);break
            if text=='/mic':
                import sounddevice as sd
                import soundfile as sf
                print('正在录制用户命令…',flush=True)
                audio=sd.rec(5*16000,samplerate=16000,channels=1,dtype='float32');sd.wait()
                path=Path(args.output)/f'command_{time.monotonic_ns()}.wav';sf.write(path,audio,16000)
                commands.put({'audio':str(path.resolve())})
            elif text:commands.put({'text':text})
    threading.Thread(target=input_commands,daemon=True).start()
    last=0
    try:
        while camera.running:
            item=camera.latest()
            if item is None or item[0]<=last:time.sleep(.02);continue
            stamp,frame=item;last=stamp
            if not commands.empty():
                cmd=commands.get()
                if cmd is None:break
                engine.emit(dict(type='guidance_paused',text='处理语音和场景问题期间暂停局部引导'))
                def latest_question_frame():
                    fresh=camera.latest()
                    return (fresh[0]-start,fresh[1]) if fresh and time.monotonic()-fresh[0]<=1.5 else None
                engine.command(cmd,frame,stamp-start,frame_provider=latest_question_frame)
                # Reacquire after potentially slow language inference; never guide on the old frame.
                stamp,frame=camera.latest();last=stamp
            view=engine.process(frame,stamp-start,capture_monotonic=stamp)
            cv2.imwrite(str(Path(args.output)/'latest.jpg'),view)
    except KeyboardInterrupt:pass
    finally:camera.close();engine.close()


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--scenario');mode.add_argument('--camera',type=int)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/demo');p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--device',default='cuda');p.add_argument('--segmentation',default='segformer-sidewalk')
    p.add_argument('--tts',action='store_true',help='Export offline Chinese response WAVs after the run; does not play them')
    args=p.parse_args()
    if args.scenario:replay(args)
    else:live(args)


if __name__=='__main__':main()
