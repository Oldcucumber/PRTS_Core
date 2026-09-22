"""Continuous core: a latest-frame vision worker and an independent speech worker.

No microphone, network service or speaker is opened implicitly. Callers supply
timestamped frames/PCM and consume structured events. The model backend is injected
so the same scheduling contract can be exercised on a CPU or mobile runtime.
"""
import json
import queue
import threading
import time
from pathlib import Path
from dataclasses import replace

from .interfaces import AudioSegmenter, SCHEMA_VERSION
from .intent import route
from .navigation import Navigation, MapUnavailable
from .tasks import Tasks


class StreamingCore:
    def __init__(self, backend, event_callback, map_provider=None, clock=time.monotonic,
                 trace_path=None, max_frame_age_s=2.0):
        self.backend, self.callback, self.clock = backend, event_callback, clock
        self.navigation = Navigation(map_provider)
        self.tasks = Tasks()
        self.max_frame_age_s = max_frame_age_s
        self.state = threading.RLock();self.event_lock=threading.Lock()
        self.frame_ready = threading.Condition()
        self.location_ready=threading.Condition();self.pending_location=None;self.location_version=0
        self.evidence_ready=threading.Condition();self.observation_version=0
        self.frame = None;self.frame_version=0;self.observation=None
        self.location = None;self.route_hint=None
        self.audio = AudioSegmenter();self.jobs=queue.Queue(maxsize=4);self.audio_jobs=queue.Queue(maxsize=4)
        self.running=False;self.workers=[];self.generation=0;self.event_sequence=0
        self.last_stop_generation=-1;self.last_wait_cancel_generation=-1
        self.playback_active=False;self.last_guide=None;self.last_guide_s=-100.
        self.last_navigation_mode=None
        self.last_route_status=None;self.last_route_notice_s=-100.
        self.stats=dict(frames_received=0, frames_processed=0, frames_replaced=0,
                        speech_jobs_dropped=0, stale_answers=0, echo_chunks_suppressed=0)
        self.trace = None
        if trace_path:
            p=Path(trace_path);p.parent.mkdir(parents=True,exist_ok=True)
            self.trace=p.open('w',encoding='utf-8')

    def emit(self, event_type, **payload):
        with self.event_lock:
            self.event_sequence += 1
            event=dict(schema_version=SCHEMA_VERSION,sequence=self.event_sequence,
                       emitted_s=self.clock(),type=event_type,**payload)
            if self.trace:
                self.trace.write(json.dumps(event,ensure_ascii=False)+'\n');self.trace.flush()
            self.callback(event)
        return event

    def say(self, text, source, priority=30, ttl_s=12, observation_s=None, cue=None):
        now=self.clock()
        event=self.emit('speech_request',text=text,source=source,priority=priority,
                        expires_s=now+ttl_s,observation_s=observation_s,
                        replace_group='navigation' if source in ('guidance','route_progress') else source)
        if cue:self.emit('sound_cue',cue=cue,related_sequence=event['sequence'],expires_s=now+ttl_s)

    def start(self):
        if self.running:return self
        self.running=True
        jobs=[('vision',self._vision_loop),('asr',self._audio_loop),('speech',self._speech_loop),
              ('location',self._location_loop)]
        if hasattr(self.backend,'collect_evidence'):jobs.append(('evidence',self._evidence_loop))
        for name,fn in jobs:
            t=threading.Thread(target=self._guard,args=(name,fn),name='prts-'+name,daemon=True)
            t.start();self.workers.append(t)
        return self

    def _guard(self,name,fn):
        try:fn()
        except Exception as e:
            import traceback
            traceback.print_exc()  # Local diagnostic log; frontend receives a short event.
            self.emit('worker_error',worker=name,error_type=type(e).__name__,text=str(e))

    def push_video(self, frame):
        with self.frame_ready:
            if self.frame is not None and frame.sequence <= self.frame.sequence:
                return False
            if self.frame is not None and frame.timestamp_s < self.frame.timestamp_s:
                return False
            self.frame=frame;self.frame_version+=1;self.stats['frames_received']+=1
            self.frame_ready.notify()
        return True

    def set_playback(self, active):
        with self.state:self.playback_active=active

    def push_audio(self, chunk):
        # Echo-clean frontends support automatic barge-in. Raw speaker feedback is
        # suppressed; a deliberate user/PTT channel is allowed to interrupt.
        if self.playback_active and chunk.role!='user' and not chunk.echo_cancelled:
            self.stats['echo_chunks_suppressed']+=1;self.audio.reset();return
        starting=self.audio.start is None
        utterance=self.audio.feed(chunk)
        if starting and (self.audio.start is not None or utterance) and chunk.role=='user':
            with self.state:
                self.generation+=1
                self.emit('speech_cancel',reason='user_speech_started',through_generation=self.generation-1)
                if hasattr(self.backend,'cancel'):self.backend.cancel()
        if utterance:
            job=dict(kind='audio',utterance=utterance,generation=self.generation)
            try:self.audio_jobs.put_nowait(job)
            except queue.Full:
                self.audio_jobs.get_nowait();self.audio_jobs.task_done()
                self.audio_jobs.put_nowait(job);self.stats['speech_jobs_dropped']+=1
                self.emit('input_dropped',input='audio_utterance',reason='asr_queue_full')

    def push_text(self, text, role='user', timestamp_s=None):
        self._enqueue(dict(kind='text',text=text,role=role,
                           timestamp_s=self.clock() if timestamp_s is None else timestamp_s))

    def _enqueue(self, job):
        is_user=job.get('role')=='user' or (job.get('kind')=='audio' and job['utterance'].role=='user')
        with self.state:
            if is_user and 'generation' not in job:
                self.generation+=1
                self.emit('speech_cancel',reason='user_input',through_generation=self.generation-1)
                if hasattr(self.backend,'cancel'):self.backend.cancel()
            job.setdefault('generation',self.generation)
        if job['kind']=='text' and job.get('role')!='ambient':
            control=route(job['text'],self.tasks.wait)
            if control and control['action'] in ('stop','cancel'):
                self._handle_text(job['text'],job['role'],job['timestamp_s'],job['generation'])
                return
        try:self.jobs.put_nowait(job)
        except queue.Full:
            old=self.jobs.get_nowait();self.jobs.task_done();self.stats['speech_jobs_dropped']+=1
            self.jobs.put_nowait(job)
            self.emit('input_dropped',input='utterance',reason='speech_queue_full')

    def update_location(self, fix):
        """Immediate already-normalized location. Frontends normally use push_location."""
        with self.state:
            self.location=fix
            event=self.navigation.update(fix,self.clock())
            if event is None:return
            self.route_hint=event if event['status']=='following' else None
            self.emit(event['type'],**{k:v for k,v in event.items() if k!='type'})
            status=(event['status'],event.get('step_index'),event.get('action'),
                    event.get('distance_to_maneuver_m',100)>20)
            if status!=self.last_route_status or self.clock()-self.last_route_notice_s>=15:
                self.say(event['text'],'route_progress',priority=50,ttl_s=6,
                         cue='arrived' if event['status']=='arrived' else None)
                self.last_route_status=status;self.last_route_notice_s=self.clock()
            if 'arrival_wait' in event:
                intent=dict(event['arrival_wait'],action='wait')
                kind,text=self.tasks.command(intent,self.clock())
                self.emit(kind,text=text,transition='navigation_to_wait')
                self.say(text,kind)

    def push_location(self,fix):
        """Latest fix input; provider coordinate conversion runs off the capture thread."""
        with self.location_ready:
            if self.pending_location and fix.timestamp_s<self.pending_location.timestamp_s:return False
            self.pending_location=fix;self.location_version+=1;self.location_ready.notify()
        return True

    def _location_loop(self):
        processed=0
        while self.running:
            with self.location_ready:
                self.location_ready.wait_for(lambda:not self.running or self.location_version!=processed,.2)
                if not self.running:return
                if self.location_version==processed:continue
                processed=self.location_version;fix=self.pending_location
            try:
                provider=self.navigation.provider
                if hasattr(provider,'convert'):
                    point=provider.convert(fix.point)
                    fix=replace(fix,point=point)
                # Do not apply an older fix after a newer one arrived during network I/O.
                with self.location_ready:
                    if processed!=self.location_version:continue
                self.update_location(fix)
            except MapUnavailable as e:
                self.emit('map_unavailable',text=str(e),operation='coordinate_conversion')
            except Exception as e:
                self.emit('request_error',worker='location',error_type=type(e).__name__,text=str(e))

    def _vision_loop(self):
        processed_version=0
        while self.running:
            with self.frame_ready:
                self.frame_ready.wait_for(lambda:not self.running or self.frame_version!=processed_version,.2)
                if not self.running:return
                if self.frame_version==processed_version:continue
                frame=self.frame
                self.stats['frames_replaced']+=max(0,self.frame_version-processed_version-1)
                processed_version=self.frame_version
            if self.clock()-frame.timestamp_s>self.max_frame_age_s:
                self.emit('frame_stale',frame_sequence=frame.sequence);continue
            with self.state:
                wait=dict(self.tasks.wait) if self.tasks.wait else None
                hint=dict(self.route_hint) if self.route_hint else None
                if self.location and self.clock()-self.location.timestamp_s>5:hint=None
            start=time.perf_counter()
            result=self.backend.observe(frame,wait=wait,route_hint=hint)
            self.stats['frames_processed']+=1
            age=self.clock()-frame.timestamp_s
            with self.state:
                self.observation=(frame,result)
                self.observation_version+=1
                with self.evidence_ready:self.evidence_ready.notify()
                self.emit('scene',frame_sequence=frame.sequence,observation_s=frame.timestamp_s,
                          processing_ms=(time.perf_counter()-start)*1000,age_s=age,
                          regions=result.get('regions',[]),detections=result.get('detections',[]),
                          guidance=result.get('guidance'),lights=result.get('lights',[]))
                guide=result.get('guidance')
                if self.tasks.navigating and guide:
                    if age>self.max_frame_age_s:
                        guide=dict(status='WAIT',direction='UNKNOWN',path=[],mode='observation_wait',reason='stale_observation',text='画面更新较慢，请稍等')
                    self._publish_guidance(guide,frame)
                if not hasattr(self.backend,'collect_evidence') and wait and self.tasks.wait and wait['version']==self.tasks.wait['version'] and age<=self.max_frame_age_s:
                    event=self.tasks.observe(result.get('ocr',[]),result.get('detections',[]),frame.timestamp_s)
                    if event:self._target(event)

    def _publish_guidance(self,guide,frame):
        mode=guide.get('mode','sidewalk')
        entered=mode=='free_forward' and self.last_navigation_mode!=mode
        key=(mode,guide['status'],guide['direction'],guide.get('reason'))
        changed=key!=self.last_guide
        if mode in ('free_forward','sidewalk'):self.last_navigation_mode=mode
        if not (changed or entered or (mode!='free_forward' and self.clock()-self.last_guide_s>=4)):return
        if guide['status']=='STOP' and changed:
            self.emit('speech_cancel',reason='navigation_hazard',preserve_wait_task=True)
        self.emit('guidance',frame_sequence=frame.sequence,observation_s=frame.timestamp_s,**guide)
        speak=mode!='free_forward' or entered or (guide['status']=='STOP' and changed)
        if speak:
            text=guide['text']
            if entered:
                text='进入自由前进模式，持续观察正前方障碍'
                if guide['status']=='STOP':text+='。'+guide['text']
            priority=(80 if guide['status'] in ('STOP','WAIT') else 40) if changed else 20
            self.say(text,'guidance',priority=priority,ttl_s=12 if entered else 3,
                     observation_s=frame.timestamp_s,cue='attention' if guide['status']=='STOP' else None)
        self.last_guide=key;self.last_guide_s=self.clock()

    def _evidence_loop(self):
        processed=0
        independent=getattr(self.backend,'independent_evidence',False)
        while self.running:
            if independent:
                with self.frame_ready:
                    self.frame_ready.wait_for(lambda:not self.running or self.frame_version!=processed,.2)
                    if not self.running:return
                    if self.frame_version==processed:continue
                    processed=self.frame_version;frame=self.frame
                with self.state:
                    if not self.tasks.wait or self.tasks.wait['notified']:continue
                    wait=dict(self.tasks.wait)
                if self.clock()-frame.timestamp_s>self.max_frame_age_s:continue
                result=self.backend.collect_frame_evidence(frame,wait)
            else:
                with self.evidence_ready:
                    self.evidence_ready.wait_for(lambda:not self.running or self.observation_version!=processed,.2)
                    if not self.running:return
                with self.state:
                    if self.observation_version==processed:continue
                    processed=self.observation_version
                    if not self.tasks.wait or self.tasks.wait['notified'] or self.observation is None:continue
                    wait=dict(self.tasks.wait);frame,scene=self.observation
                result=self.backend.collect_evidence(frame,scene,wait)
            with self.state:
                if not self.tasks.wait or wait['version']!=self.tasks.wait['version']:continue
                age=self.clock()-frame.timestamp_s
                self.emit('target_evidence',frame_sequence=frame.sequence,observation_s=frame.timestamp_s,
                          age_s=age,ocr=result.get('ocr',[]),detections=result.get('detections',[]),
                          wait_version=wait['version'])
                if 'brain_decision' in result:
                    decision=result['brain_decision']
                    evidence_s=decision.get('evidence_s',frame.timestamp_s);age=self.clock()-evidence_s
                    self.emit('brain_observation',frame_sequence=frame.sequence,observation_s=evidence_s,
                              age_s=age,**decision)
                    # A slower multimodal turn describes a past observation, never
                    # a current path or permission to move. Keep its clock explicit.
                    if age>(45 if wait['kind']=='number' else 30):continue
                    event=self.tasks.apply_brain(decision,evidence_s)
                    if event:
                        event['evidence_age_s']=age
                        if age>self.max_frame_age_s:
                            channel='环境广播' if decision.get('observation_channel')=='environment_transcript' else '画面'
                            event['text']=f'约{max(1,round(age))}秒前的{channel}中，'+event['text']
                        self._target(event)
                    continue
                # A retrospective scene observation is useful for waiting. It is
                # not a current movement instruction; lights need a shorter budget.
                max_age=1.5 if wait['kind']=='light' else 6.
                if age>max_age:continue
                event=self.tasks.observe(result.get('ocr',[]),result.get('detections',[]),frame.timestamp_s)
                if event:
                    event['evidence_age_s']=age
                    self._target(event)

    def _target(self,event):
        self.emit(event['type'],**{k:v for k,v in event.items() if k!='type'})
        confirmed=event['type']=='target_observed'
        self.say(event['text'],event['type'],priority=70 if confirmed else 30,ttl_s=12,
                 cue='target_found' if confirmed else None)

    def _audio_loop(self):
        while self.running or not self.audio_jobs.empty():
            try:job=self.audio_jobs.get(timeout=.2)
            except queue.Empty:continue
            try:
                u=job['utterance'];result=self.backend.transcribe(u.samples,role=u.role)
                self.emit('transcript',text=result['text'],role=u.role,observation_s=u.timestamp_s,end_s=u.end_s,
                          processing_ms=result.get('processing_ms'))
                if result['text'].strip():
                    self._enqueue(dict(kind='text',text=result['text'],role=u.role,timestamp_s=u.timestamp_s,
                                       generation=job['generation']))
            except Exception as e:
                self.emit('request_error',worker='asr',error_type=type(e).__name__,text=str(e))
            finally:self.audio_jobs.task_done()

    def _speech_loop(self):
        while self.running or not self.jobs.empty() or not self.audio_jobs.empty() or self.audio_jobs.unfinished_tasks:
            try:job=self.jobs.get(timeout=.2)
            except queue.Empty:continue
            try:
                text=job['text'];stamp=job['timestamp_s'];role=job['role']
                if text.strip():self._handle_text(text,role,stamp,job['generation'])
            except MapUnavailable as e:
                self.emit('map_unavailable',text=str(e));self.say(str(e),'map_unavailable')
            except Exception as e:
                self.emit('request_error',error_type=type(e).__name__,text=str(e))
            finally:self.jobs.task_done()

    def _handle_text(self,text,role,stamp,generation):
        with self.state:
            intent=route(text,self.tasks.wait,bool(self.navigation.candidates)) if role!='ambient' else None
            if role=='ambient' or (role=='auto' and intent is None):
                if getattr(self.backend,'brain_waiting',False):
                    self.backend.observe_ambient(text,stamp)
                    return
                event=self.tasks.observe([],[],self.clock(),audio_text=text,audio_start=stamp)
                if event:self._target(event)
                return
        if intent is None:intent=self.backend.intent(text)
        action=intent['action']
        with self.state:
            if ((action in ('navigate','destination','confirm_destination') and generation<self.last_stop_generation)
                    or (action in ('wait','continue_wait') and generation<self.last_wait_cancel_generation)):
                self.emit('request_superseded',observation_s=stamp);return
        self.emit('intent',utterance=text,observation_s=stamp,**intent)
        if action=='destination':
            event=self.navigation.search(intent['query'],self.location,intent.get('arrival_wait'))
            if generation!=self.generation:
                self.navigation.cancel();self.emit('request_superseded',observation_s=stamp);return
            self.emit(event['type'],**{k:v for k,v in event.items() if k!='type'})
            self.say(event['text'],event['type']);return
        if action=='confirm_destination':
            index=intent['index']
            if index==0 and len(self.navigation.candidates)==1:index=1
            if self.location is None:
                self.say('需要当前位置才能规划路线','location_required');return
            event=self.navigation.confirm(index,self.location.point)
            if generation!=self.generation:
                self.navigation.cancel();self.emit('request_superseded',observation_s=stamp);return
            with self.state:
                if event['type']=='route_started':self.tasks.navigating=True
            self.emit(event['type'],**{k:v for k,v in event.items() if k!='type'})
            self.say(event['text'],event['type']);return
        with self.state:
            if action in ('navigate','stop'):
                self.last_navigation_mode=None;self.last_guide=None
            if action=='stop':
                self.last_stop_generation=max(self.last_stop_generation,generation)
                self.navigation.cancel();self.route_hint=None
            if action=='cancel':self.last_wait_cancel_generation=max(self.last_wait_cancel_generation,generation)
            kind,message=self.tasks.command(intent,self.clock())
            self.emit(kind,text=message,observation_s=stamp)
            if action!='ask':self.say(message,kind);return
            if generation!=self.generation:
                self.emit('request_superseded',observation_s=stamp);return
            with self.frame_ready:frame=self.frame
            context=self.observation
        if frame is None or self.clock()-frame.timestamp_s>self.max_frame_age_s:
            self.say('当前没有新鲜画面，请调整相机后再问','answer');return
        answer=self.backend.answer(text,frame,context[1] if context and context[0].sequence==frame.sequence else None)
        with self.state:
            if generation!=self.generation:
                self.stats['stale_answers']+=1
                self.emit('answer_cancelled',reason='superseded',frame_sequence=frame.sequence);return
            age=self.clock()-frame.timestamp_s
            retrospective=age>self.max_frame_age_s
            self.emit('answer',question=text,frame_sequence=frame.sequence,observation_s=frame.timestamp_s,
                      evidence_age_s=age,retrospective=retrospective,**answer)
            spoken=(f'根据约{max(1,round(age))}秒前的画面，'+answer['text']) if retrospective else answer['text']
            self.say(spoken,'answer',observation_s=frame.timestamp_s,ttl_s=max(12,min(60,len(spoken)/3+5)))

    def close(self, timeout_s=30):
        self.running=False
        with self.frame_ready:self.frame_ready.notify_all()
        with self.evidence_ready:self.evidence_ready.notify_all()
        with self.location_ready:self.location_ready.notify_all()
        deadline=time.monotonic()+timeout_s
        for t in self.workers:t.join(max(0,deadline-time.monotonic()))
        alive=[t.name for t in self.workers if t.is_alive()]
        if alive:raise TimeoutError('Core workers did not finish: '+', '.join(alive))
        self.emit('session_ended',stats=dict(self.stats))
        if self.trace:self.trace.close()
