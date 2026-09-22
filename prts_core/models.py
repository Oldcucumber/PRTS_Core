"""Real pretrained local models. Downloads belong to scripts/download_models.py."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
import time
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
runtime_config=ROOT/'outputs/runtime/ultralytics'
runtime_config.mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(runtime_config)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


class Perception:
    def __init__(self, model_dir=ROOT/'models', device='cuda', long_side=512, detector_size=640, segmentation='segformer-cityscapes', square=True, precision='fp32'):
        from transformers import SegformerForSemanticSegmentation,Mask2FormerForUniversalSegmentation
        from ultralytics import YOLO
        from .semantics import WALKABLE
        self.device, self.long_side, self.detector_size = device, long_side, detector_size
        if precision not in ('fp32','fp16'):raise ValueError('precision must be fp32 or fp16')
        if precision=='fp16' and not str(device).startswith('cuda'):raise ValueError('Use fp32 for CPU; fp16 is an optional CUDA optimization')
        self.precision=precision
        self.square=square
        if segmentation.endswith('-sem.pt'):
            path=Path(model_dir)/segmentation
            if not path.exists():raise FileNotFoundError(path)
            wrapper=YOLO(str(path),task='semantic');self.seg_kind='yolo_semantic'
            self.seg=wrapper.model.eval()
            self.seg=(self.seg.half() if precision=='fp16' else self.seg.float()).to(device);self.labels=wrapper.names
        else:
            self.seg_kind=json.loads((Path(model_dir)/segmentation/'config.json').read_text())['model_type']
            model_class=Mask2FormerForUniversalSegmentation if self.seg_kind=='mask2former' else SegformerForSemanticSegmentation
            self.seg = model_class.from_pretrained(
                str(Path(model_dir)/segmentation), local_files_only=True,
                dtype=torch.float16 if precision=='fp16' else torch.float32).eval().to(device)
            self.labels=self.seg.config.id2label
        self.sidewalk_ids=[k for k,v in self.labels.items() if v.casefold() in WALKABLE]
        self.road_id=next(k for k,v in self.labels.items() if v.casefold() in ('road','flat-road'))
        weight = Path(model_dir)/'yolo11n.pt'
        if not weight.is_file(): raise FileNotFoundError(weight)
        self.det = YOLO(str(weight))
        self.mean = np.array([.485,.456,.406], np.float32)
        self.std = np.array([.229,.224,.225], np.float32)

    def __call__(self, bgr, detect=True):
        h,w=bgr.shape[:2]
        scale=self.long_side/max(h,w)
        iw=max(32,round(w*scale/32)*32); ih=max(32,round(h*scale/32)*32)
        if self.square:iw=ih=self.long_side
        rgb=cv2.cvtColor(cv2.resize(bgr,(iw,ih)),cv2.COLOR_BGR2RGB)
        x=rgb.astype(np.float32)/255
        if self.seg_kind!='yolo_semantic':x=(x-self.mean)/self.std
        x=torch.from_numpy(x.transpose(2,0,1).copy()[None]).to(self.device)
        if self.precision=='fp16':x=x.half()
        sync(self.device);start=time.perf_counter()
        with torch.inference_mode():
            gh=round(h*min(1,320/max(h,w)));gw=round(w*min(1,320/max(h,w)))
            if self.seg_kind=='mask2former':
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(self.device).startswith('cuda')):
                    result=self.seg(pixel_values=x)
                class_probs=result.class_queries_logits.float().softmax(-1)[...,:-1]
                masks=result.masks_queries_logits.float()
                masks=torch.nn.functional.interpolate(masks,size=(gh,gw),mode='bilinear',align_corners=False).sigmoid()
                scores=torch.einsum('bqc,bqhw->bchw',class_probs,masks)[0]
                probs=scores/scores.sum(0,keepdim=True).clamp_min(1e-8)
            else:
                if self.seg_kind=='yolo_semantic':
                    logits=self.seg(x)
                    if isinstance(logits,(list,tuple)):logits=logits[0]
                else:logits=self.seg(pixel_values=x).logits
                probs=torch.nn.functional.interpolate(logits,size=(gh,gw),mode='bilinear',align_corners=False).softmax(1)[0]
            classes=probs.argmax(0).cpu().numpy().astype(np.uint8)
            confidence=probs.max(0).values.cpu().numpy()
            sidewalk=probs[self.sidewalk_ids].sum(0).cpu().numpy();road=probs[self.road_id].cpu().numpy()
            sidewalk_class=np.isin(classes,self.sidewalk_ids)
        sync(self.device);seg_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        detections=[]
        if detect:
            prediction=self.det.predict(bgr,imgsz=self.detector_size,conf=.25,iou=.55,device=self.device,
                                        verbose=False,save=False)[0]
            for box in prediction.boxes:
                cid=int(box.cls.item());xy=box.xyxyn[0].cpu().tolist()
                detections.append(dict(box=[min(1,max(0,float(v))) for v in xy],
                                       score=float(box.conf.item()),class_id=cid,label=self.det.names[cid],
                                       blocking=cid not in (9,11),approaching=False))
        sync(self.device)
        det_ms=(time.perf_counter()-start)*1000
        from .semantics import group_labels,region_records
        return dict(classes=classes,semantic_groups=group_labels(classes,self.labels),labels=self.labels,
                    confidence=confidence,regions=region_records(classes,self.labels),
                    sidewalk_class=sidewalk_class,sidewalk=sidewalk,road=road,detections=detections,
                    input_size=[iw,ih],coordinate_space='image_normalized',
                    timings={'seg_ms':seg_ms,'det_ms':det_ms})


class OCR:
    def __init__(self):
        import rapidocr
        root=Path(rapidocr.__file__).parent/'models'
        files={'Det.model_path':root/'PP-OCRv6_det_small.onnx',
               'Rec.model_path':root/'PP-OCRv6_rec_small.onnx',
               'Cls.model_path':root/'ch_ppocr_mobile_v2.0_cls_mobile.onnx'}
        for f in files.values():
            if not f.exists(): raise FileNotFoundError(f'Install rapidocr==3.9.2 with bundled OCR weights: {f}')
        self.engine=rapidocr.RapidOCR(params={**{k:str(v) for k,v in files.items()},
           'Global.log_level':'warning','EngineConfig.onnxruntime.intra_op_num_threads':4,
           'EngineConfig.onnxruntime.inter_op_num_threads':1,'Det.limit_side_len':960})

    def __call__(self,bgr):
        result=self.engine(bgr)
        if result.txts is None: return []
        return [dict(text=str(t),score=float(s),box=np.asarray(b).tolist())
                for t,s,b in zip(result.txts,result.scores,result.boxes)]


class ASR:
    def __init__(self, model_dir=ROOT/'models', variant='medium'):
        from faster_whisper import WhisperModel
        p=Path(model_dir)/('whisper-'+variant)
        if not (p/'model.bin').is_file(): raise FileNotFoundError(p)
        self.variant=variant
        self.model=WhisperModel(str(p),device='cpu',compute_type='int8',cpu_threads=6,local_files_only=True)

    def __call__(self,audio,role='user'):
        start=time.perf_counter()
        prompt=('这是用户关于公交车、地铁、人行道导航、叫号提醒和读牌的语音命令。' if role=='user'
                else '这是一段环境广播。公交车到站、列车进站、叫号与窗口通知。')
        segments,info=self.model.transcribe(str(audio) if isinstance(audio,(str,Path)) else audio,
            language='zh',beam_size=5,vad_filter=True,condition_on_previous_text=False,initial_prompt=prompt)
        segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,
                       no_speech_prob=s.no_speech_prob) for s in segments]
        return dict(text=''.join(s['text'] for s in segments),segments=segments,model='whisper-'+self.variant,context_prompt=prompt,
                    processing_ms=(time.perf_counter()-start)*1000)


class Brain:
    def __init__(self,model_dir=ROOT/'models',device='cuda'):
        from transformers import AutoModelForImageTextToText,AutoProcessor
        path=Path(model_dir)/'minicpm-v-4.6'
        if not (path/'model.safetensors').is_file():raise FileNotFoundError(path/'model.safetensors')
        self.device=device
        self.processor=AutoProcessor.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModelForImageTextToText.from_pretrained(str(path),local_files_only=True,
            dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32,
            attn_implementation='sdpa').eval().to(device)

    def generate(self,question,bgr=None,context='',max_tokens=180,raw_prompt=False):
        from PIL import Image
        content=[]
        if bgr is not None:
            content.append({'type':'image','image':Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))})
        prompt=('你是PRTS离线视觉助手。用简洁中文回答用户具体问题。可见文字照实读，不猜测被遮挡内容；'
                '读告示时先按阅读顺序转写，再解释。必须保留原文的适用主体、否定和条件，'
                '不得把针对车辆的限制扩大到行人，不把文字摘要变成行动指令。'
                '区分人行道与车行道。不提供未经测量的米数，不承诺安全过街。'+
                (f'\n其他感知结果（可能有误，仅供核对）：{context}' if context else '')+'\n用户：'+question)
        if raw_prompt:prompt=question
        content.append({'type':'text','text':prompt})
        messages=[{'role':'user','content':content}]
        start=time.perf_counter()
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt',processor_kwargs={'downsample_mode':'4x','max_slice_nums':4}).to(self.device)
        with torch.inference_mode():
            generated=self.model.generate(**inputs,downsample_mode='4x',max_new_tokens=max_tokens,
                                          do_sample=False)
        sync(self.device)
        text=self.processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        return dict(text=text,processing_ms=(time.perf_counter()-start)*1000,
                    input_tokens=int(inputs.input_ids.shape[1]),output_tokens=int(generated.shape[1]-inputs.input_ids.shape[1]))

    def intent(self,text):
        prompt=('把下面的用户话语分类，只输出JSON，不回答场景问题。'
                'action只能是navigate（开始或继续局部导航）、stop（停止导航）、ask（询问眼前信息）、'
                'wait（等目标出现时提醒）、cancel（取消等待）。'
                'wait时kind必须从以下单个英文值选择：bus表示公交车，train表示列车或地铁，'
                'number表示叫号，light表示交通灯。绝不输出斜杠或多个值。target是线路号/叫号号码/灯色，'
                'target只提取当前用户实际给出的标识，中文数字转阿拉伯数字，保留字母和前导零；'
                '不得沿用示例、历史视频或默认线路，未提供标识时留空。其他action的kind和target为空。'
                'JSON格式：{"action":"ask","kind":"","target":""}。\n用户话语：'+text)
        prompt+='\n示例：帮我沿着人行道往前走 -> {"action":"navigate","kind":"","target":""}'
        prompt+='\n等待实体规则：公交线路对应bus，地铁/列车线路对应train，排队叫号对应number。'
        prompt+='\n用户给出什么线路或号码就提取什么；改变目标时以本次话语为准。'
        prompt+='\n示例：牌子写的什么 -> {"action":"ask","kind":"","target":""}'
        prompt+='\n现在只处理用户话语：'+text
        result=self.generate(prompt,max_tokens=100,raw_prompt=True)
        match=re.search(r'\{[^{}]*\}',result['text'])
        try:
            intent=json.loads(match.group(0)) if match else {}
        except ValueError:intent={}
        if intent.get('action') not in ('navigate','stop','ask','wait','cancel'):
            intent={'action':'ask','kind':'','target':'','parse_fallback':True}
        if intent['action']!='wait':intent.update(kind='',target='')
        else:intent['target']=str(intent.get('target') or '').strip()
        intent['raw_model']=result['text'];intent['processing_ms']=result['processing_ms']
        return intent
