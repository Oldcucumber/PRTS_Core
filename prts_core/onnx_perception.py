"""Portable CPU perception. Imports no PyTorch, CUDA or TensorRT."""
import json,time,threading
from contextlib import nullcontext
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from .semantics import WALKABLE,group_labels,region_records


def letterbox(bgr,width,height):
    h,w=bgr.shape[:2];scale=min(width/w,height/h);rw,rh=round(w*scale),round(h*scale)
    x,y=(width-rw)//2,(height-rh)//2
    canvas=np.full((height,width,3),114,np.uint8)
    canvas[y:y+rh,x:x+rw]=cv2.resize(bgr,(rw,rh))
    return canvas,(x,y,rw,rh)


class ONNXPerception:
    def __init__(self,semantic_dir,detector_dir=None,threads=6,low_memory=False,providers=None):
        folder=Path(semantic_dir);self.meta=json.loads((folder/'manifest.json').read_text())
        opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        providers=providers or ['CPUExecutionProvider']
        if any(p not in ort.get_available_providers() for p in providers):raise ValueError('Requested ONNX provider unavailable')
        if 'DmlExecutionProvider' in providers:
            opts.enable_mem_pattern=False;opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.runtime='onnxruntime_'+providers[0].removesuffix('ExecutionProvider').lower()
        self.det_lock=threading.Lock()
        # This tested Windows DML build failed under simultaneous segmentation
        # and detector Run calls. Serialize only its GPU dispatches; OCR, ASR,
        # language inference and camera ingestion keep their own workers.
        self.seg_lock=self.det_lock if 'DmlExecutionProvider' in providers else nullcontext()
        if low_memory:
            opts.enable_cpu_mem_arena=False;opts.enable_mem_pattern=False
            opts.add_session_config_entry('session.disable_prepacking','1')
        provider_options=[{'device_id':'0'} if p=='DmlExecutionProvider' else {} for p in providers]
        self.seg=ort.InferenceSession(str(folder/'semantic.onnx'),opts,providers=providers,provider_options=provider_options)
        self.labels={int(k):v for k,v in self.meta['labels'].items()};self.shape=self.meta['input_shape']
        self.sidewalk_ids=[k for k,v in self.labels.items() if v.casefold() in WALKABLE]
        self.road_ids=[k for k,v in self.labels.items() if v.casefold() in ('road','flat-road')]
        self.det=None
        if detector_dir:
            folder=Path(detector_dir);self.det_meta=json.loads((folder/'manifest.json').read_text())
            self.det=ort.InferenceSession(str(folder/'detector.onnx'),opts,providers=providers,provider_options=provider_options)

    def __call__(self,bgr,detect=True):
        start=time.perf_counter();h,w=bgr.shape[:2];ih,iw=self.shape[2:]
        canvas,crop=letterbox(bgr,iw,ih)
        rgb=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if self.meta['mean'] is not None:rgb=(rgb-np.asarray(self.meta['mean'],np.float32))/np.asarray(self.meta['std'],np.float32)
        x=np.ascontiguousarray(rgb.transpose(2,0,1)[None])
        with self.seg_lock:scores=self.seg.run(None,{'rgb':x})[0][0]
        # Decode logits/probabilities in their original output grid, crop padding,
        # and resize to the caller's image aspect ratio without stretching masks.
        x0,y0,rw,rh=crop;sh,sw=scores.shape[1:]
        xa,ya=round(x0*sw/iw),round(y0*sh/ih);xb,yb=round((x0+rw)*sw/iw),round((y0+rh)*sh/ih)
        scores=scores[:,ya:yb,xa:xb];gh=round(h*min(1,320/max(h,w)));gw=round(w*min(1,320/max(h,w)))
        scores=cv2.resize(scores.transpose(1,2,0),(gw,gh),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
        if 'probabilities' not in self.meta['output']:
            scores=np.exp(scores-scores.max(0,keepdims=True))
        probs=scores/np.maximum(scores.sum(0,keepdims=True),1e-8)
        classes=probs.argmax(0).astype(np.uint8);sidewalk=probs[self.sidewalk_ids].sum(0);road=probs[self.road_ids].sum(0)
        seg_ms=(time.perf_counter()-start)*1000;start=time.perf_counter()
        detections=self.detect(bgr) if detect else []
        det_ms=(time.perf_counter()-start)*1000
        return {'classes':classes,'labels':self.labels,'semantic_groups':group_labels(classes,self.labels),
                'confidence':probs.max(0),'sidewalk_class':np.isin(classes,self.sidewalk_ids),'sidewalk':sidewalk,'road':road,
                'regions':region_records(classes,self.labels),'detections':detections,'coordinate_space':'image_normalized',
                'timings':{'seg_ms':seg_ms,'det_ms':det_ms},'runtime':self.runtime,'input_size':[iw,ih]}

    def detect(self,bgr):
        """Fast current-frame objects, also usable while dense ground inference runs."""
        detections=[]
        if self.det is not None:
            dh,dw=self.det_meta['input_shape'][2:];canvas,crop=letterbox(bgr,dw,dh)
            x=np.ascontiguousarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None].astype(np.float32)/255)
            # DirectML forbids concurrent Run on the same session. Ground and
            # waiting share this detector, while their other sessions can run.
            with self.det_lock:raw=self.det.run(None,{'rgb':x})[0][0].T
            confidence=raw[:,4:].max(1);cids=raw[:,4:].argmax(1);chosen=confidence>=.25;raw=raw[chosen];confidence=confidence[chosen];cids=cids[chosen]
            boxes=raw[:,:4].copy();boxes[:,:2]-=boxes[:,2:]/2
            # Class-aware NMS, matching the reference detector thresholds.
            keep=[]
            for cid in np.unique(cids):
                idx=np.flatnonzero(cids==cid);kept=cv2.dnn.NMSBoxes(boxes[idx].tolist(),confidence[idx].tolist(),.25,.55)
                keep.extend(idx[np.asarray(kept,dtype=int).ravel()].tolist())
            ox,oy,rw,rh=crop
            for idx in sorted(keep,key=lambda i:-confidence[i]):
                x,y,bw,bh=boxes[idx];cid=int(cids[idx]);box=[(x-ox)/rw,(y-oy)/rh,(x+bw-ox)/rw,(y+bh-oy)/rh]
                detections.append({'box':[float(np.clip(v,0,1)) for v in box],'score':float(confidence[idx]),
                    'class_id':cid,'label':self.det_meta['labels'][str(cid)],'blocking':cid not in (9,11),'approaching':False})
        return detections
