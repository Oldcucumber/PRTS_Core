"""Real local model adapter for the streaming reference core."""
import inspect
import json
import threading
import time
from collections import deque
import cv2
import numpy as np

from .intent import ordered_ocr, read_text_request


class LocalBackend:
    def __init__(self, model_dir, device='cpu', segmentation='yolo26s-sem.pt',
                 long_side=768, asr_variant='medium', prewarm_audio=False,
                 native_library=None,precision='fp32',onnx_semantic_dir=None,onnx_detector_dir=None,
                 onnx_provider='CPUExecutionProvider',native_language='Q5_K_M',ocr_profile='bounded_upright',
                 native_projector='mmproj-model-f16.gguf',waiting_mode='brain',
                 native_model_subdir='minicpm-v-4.6-gguf',native_language_file=None,
                 native_server=None,native_server_log=None,native_sampling='recommended',brain_strategy='direct',native_threads=4):
        from .guidance import Guide
        self.model_dir,self.device,self.asr_variant=model_dir,device,asr_variant
        self.native_library=native_library
        self.native_language=native_language
        self.ocr_profile=ocr_profile
        self.native_projector=native_projector
        self.native_model_subdir=native_model_subdir
        self.native_language_file=native_language_file or f'MiniCPM-V-4_6-{native_language}.gguf'
        self.native_server=native_server;self.native_server_log=native_server_log
        self.native_sampling=native_sampling;self.brain_strategy=brain_strategy
        self.native_threads=native_threads
        self.waiting_mode=waiting_mode
        self.brain_waiting=waiting_mode=='brain'
        self.independent_evidence=self.brain_waiting or bool(onnx_semantic_dir)
        self._waiting_brain=None;self.ambient=deque(maxlen=4);self.last_guidance=None
        self.ambient_lock=threading.Lock()
        if onnx_semantic_dir:
            from .onnx_perception import ONNXPerception
            self.perception=ONNXPerception(onnx_semantic_dir,onnx_detector_dir,threads=4,low_memory=True,providers=[onnx_provider])
        else:
            from .models import Perception
            if device=='cpu':
                import torch
                torch.set_num_threads(4)
            self.perception=Perception(model_dir,device,long_side=long_side,segmentation=segmentation,square=False,precision=precision)
        self.guide=Guide()
        self.accepts_route='route_hint' in inspect.signature(self.guide.__call__).parameters
        self._ocr=None;self._asr=None;self._brain=None
        self.ocr_lock=threading.Lock();self.brain_lock=threading.Lock();self.last_view=None
        if prewarm_audio:self._load_asr()

    def _load_asr(self):
        if self._asr is None:
            if self.asr_variant=='sensevoice':
                from .asr_sensevoice import SenseVoiceASR
                self._asr=SenseVoiceASR(self.model_dir)
            else:
                from .models import ASR
                self._asr=ASR(self.model_dir,self.asr_variant)
        return self._asr

    def _load_brain(self):
        with self.brain_lock:
            if self._brain is None:
                if self.native_server:
                    from pathlib import Path
                    from .server_vlm import ServerVLM
                    root=Path(self.model_dir)/self.native_model_subdir
                    self._brain=ServerVLM(self.native_server,root/self.native_language_file,root/self.native_projector,
                        self.native_server_log or Path('outputs/runtime')/f'brain-{time.time_ns()}.log',sampling=self.native_sampling)
                elif self.native_library:
                    from .native_vlm import NativeVLM
                    self._brain=NativeVLM(self.model_dir,self.native_library,image_slices=1,threads=self.native_threads,
                        language_file=self.native_language_file,projector_file=self.native_projector,
                        model_subdir=self.native_model_subdir,sampling=self.native_sampling)
                else:
                    from .models import Brain
                    self._brain=Brain(self.model_dir,self.device)
        return self._brain

    def read(self, bgr):
        with self.ocr_lock:
            if self._ocr is None:
                from .ocr import LocalOCR
                from pathlib import Path
                ocr_dir=Path(self.model_dir)/'ocr'
                self._ocr=LocalOCR(profile=self.ocr_profile,model_dir=ocr_dir if ocr_dir.is_dir() else None)
            return self._ocr(bgr)

    def transcribe(self,samples,role='user'):
        return self._load_asr()(samples,role='ambient' if role=='ambient' else 'user')

    def intent(self,text):
        return self._load_brain().intent(text)

    def observe(self,frame,wait=None,route_hint=None):
        from .guidance import draw
        from .semantics import semantic_view
        result=self.perception(frame.bgr)
        kwargs={'route_hint':route_hint} if self.accepts_route else {}
        guide=self.guide(frame.bgr,result,frame.timestamp_s,**kwargs)
        for i,d in enumerate(result['detections']):d['observation_id']=f'{frame.sequence}:{i}'
        clean={k:v for k,v in guide.items() if not k.startswith('_')}
        self.last_guidance=dict(status=clean['status'],direction=clean['direction'],observation_s=frame.timestamp_s)
        # The audit preview does not need camera resolution. Keep original pixels
        # for perception/OCR, but avoid several 4K float buffers for decoration.
        h,w=frame.bgr.shape[:2];scale=min(1.,960/max(h,w))
        preview=cv2.resize(frame.bgr,(round(w*scale),round(h*scale))) if scale<1 else frame.bgr
        self.last_view=(frame.sequence,frame.timestamp_s,draw(semantic_view(preview,result,legend=False),guide))
        return dict(guidance=clean,regions=result.get('regions',[]),detections=result['detections'],
                    timings=result['timings'])

    def collect_evidence(self,frame,scene,wait):
        if self.brain_waiting:return self.collect_frame_evidence(frame,wait)
        from .vehicle_evidence import read_vehicle,verify_with_vlm
        ocr=[];h,w=frame.bgr.shape[:2]
        if wait['kind'] in ('bus','train'):
            for d in scene['detections']:
                if d['label']!=wait['kind']:continue
                verifier=(lambda crop:verify_with_vlm(self._load_brain(),crop,wait['kind'])) if self.native_library else None
                ocr.extend(read_vehicle(frame.bgr,d,self.read,verifier))
        elif wait['kind']=='number':
            from .queue_evidence import read_queue
            ocr=read_queue(frame.bgr,self.read)
        return dict(ocr=ocr,detections=scene['detections'])

    def collect_frame_evidence(self,frame,wait):
        if self.brain_waiting:
            from .waiting_brain import WaitingBrain
            if self._waiting_brain is None:self._waiting_brain=WaitingBrain(self._load_brain(),strategy=self.brain_strategy)
            ocr=self.read(frame.bgr)
            with self.ambient_lock:
                ambient=[x for x in self.ambient if wait['started']<=x['observation_s'] and 0<=frame.timestamp_s-x['observation_s']<=45]
            decision=self._waiting_brain.observe(frame.bgr,wait,frame.timestamp_s,ocr,ambient,self.last_guidance)
            return dict(ocr=ocr,detections=[],brain_decision=decision)
        detections=self.perception.detect(frame.bgr)
        for i,d in enumerate(detections):d['observation_id']=f'{frame.sequence}:{i}'
        return self.collect_evidence(frame,dict(detections=detections),wait)

    def observe_ambient(self,text,timestamp_s):
        with self.ambient_lock:self.ambient.append(dict(text=text,observation_s=timestamp_s))

    def cancel(self):
        if self._brain is not None and hasattr(self._brain,'cancel'):self._brain.cancel()

    def answer(self,text,frame,context=None):
        start=time.perf_counter();ocr=self.read(frame.bgr);lines=ordered_ocr(ocr)
        if read_text_request(text) and lines:
            return dict(text='画面可辨认的文字：'+'；'.join(lines)+'。',source='ocr_transcription',
                        ocr=ocr,processing_ms=(time.perf_counter()-start)*1000)
        evidence={'当前帧文字':lines,'当前帧检测':context.get('detections',[]) if context else [],
                  '提示':'画面和证据描述当前观测，不推断未观测到的通行许可。'}
        result=self._load_brain().generate(text,frame.bgr,context=json.dumps(evidence,ensure_ascii=False))
        return dict(result,source='local_vision_language_model',ocr=ocr)
