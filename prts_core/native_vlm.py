"""ctypes binding of the same C ABI supplied to Swift, no PyTorch or CUDA."""
import codecs
import ctypes as C
import json
from pathlib import Path
import re
import threading
import time
import cv2
import numpy as np


class Config(C.Structure):
    _fields_=[('model_path',C.c_char_p),('projector_path',C.c_char_p),('context_tokens',C.c_int32),
              ('threads',C.c_int32),('image_slices',C.c_int32),('use_gpu',C.c_int32)]


Callback=C.CFUNCTYPE(None,C.POINTER(C.c_uint8),C.c_size_t,C.c_void_p)


class NativeVLM:
    def __init__(self,model_dir,library,threads=4,image_slices=4,use_gpu=False,
                 language_file='MiniCPM-V-4_6-Q5_K_M.gguf',projector_file='mmproj-model-f16.gguf',model_subdir='minicpm-v-4.6-gguf',sampling='recommended'):
        self.lib=C.CDLL(str(Path(library).resolve()))
        self.lib.prts_vlm_create.argtypes=[C.POINTER(Config)];self.lib.prts_vlm_create.restype=C.c_void_p
        self.lib.prts_vlm_run.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_uint8),C.c_uint32,C.c_uint32,C.c_int32,Callback,C.c_void_p]
        self.lib.prts_vlm_run.restype=C.c_int32
        self.lib.prts_vlm_runtime_revision.restype=C.c_char_p
        self.runtime_revision=self.lib.prts_vlm_runtime_revision().decode('utf-8')
        self.supports_json_schema=hasattr(self.lib,'prts_vlm_run_json') and self.runtime_revision.startswith('llama.cpp@')
        if self.supports_json_schema:
            self.lib.prts_vlm_run_json.argtypes=[C.c_void_p,C.c_char_p,C.c_char_p,C.POINTER(C.c_uint8),C.c_uint32,C.c_uint32,C.c_int32,Callback,C.c_void_p]
            self.lib.prts_vlm_run_json.restype=C.c_int32
        self.lib.prts_vlm_cancel.argtypes=[C.c_void_p];self.lib.prts_vlm_cancel.restype=None
        self.lib.prts_vlm_destroy.argtypes=[C.c_void_p];self.lib.prts_vlm_destroy.restype=None
        self.lib.prts_vlm_last_error.restype=C.c_char_p
        root=Path(model_dir)/model_subdir
        self.model_name=Path(language_file).stem+'+'+Path(projector_file).stem
        config=Config(str(root/language_file).encode('utf-8'),
                      str(root/projector_file).encode('utf-8'),4096,threads,image_slices,int(use_gpu))
        self.handle=self.lib.prts_vlm_create(C.byref(config));self.lock=threading.Lock()
        if not self.handle:raise RuntimeError(self.lib.prts_vlm_last_error().decode('utf-8'))
        self.sampling=sampling
        if sampling=='greedy':
            self.lib.prts_vlm_set_deterministic.argtypes=[C.c_void_p,C.c_int32]
            self.lib.prts_vlm_set_deterministic.restype=None
            self.lib.prts_vlm_set_deterministic(self.handle,1)

    def generate(self,question,bgr=None,context='',max_tokens=180,raw_prompt=False,on_delta=None,response_schema=None):
        prompt=question if raw_prompt else ('你是PRTS离线视觉助手。用简洁中文回答具体问题；只描述可见信息，'
            '保留文字的否定、适用主体和条件，不猜测遮挡内容或未测量距离，不把信号灯观测当作通行许可。'
            +(f'\n其他感知证据，仅供核对：{context}' if context else '')+'\n用户：'+question)
        rgb=None if bgr is None else np.ascontiguousarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        data=None if rgb is None else rgb.ctypes.data_as(C.POINTER(C.c_uint8))
        h,w=(0,0) if rgb is None else rgb.shape[:2]
        decoder=codecs.getincrementaldecoder('utf-8')();pieces=[]
        @Callback
        def callback(buf,n,_):
            value=decoder.decode(C.string_at(buf,n))
            pieces.append(value)
            if on_delta and value:on_delta(value)
        start=time.perf_counter()
        with self.lock:
            if not self.handle:raise RuntimeError('Native VLM is closed')
            if response_schema is not None:
                code=self.lib.prts_vlm_run_json(self.handle,prompt.encode('utf-8'),json.dumps(response_schema,ensure_ascii=False).encode('utf-8'),
                    data,w,h,max_tokens,callback,None)
            else:code=self.lib.prts_vlm_run(self.handle,prompt.encode('utf-8'),data,w,h,max_tokens,callback,None)
        if code<0:raise RuntimeError(self.lib.prts_vlm_last_error().decode('utf-8'))
        # Cancellation/output limit may split a UTF-8 token; keep the complete prefix.
        return dict(text=''.join(pieces).strip(),processing_ms=(time.perf_counter()-start)*1000,
                    finish_reason={0:'stop',1:'cancelled',2:'length'}[code],model=self.model_name,runtime=self.runtime_revision,sampling=self.sampling)

    def intent(self,text):
        prompt=('提取用户的任务为JSON，字段action为ask、navigate、stop、wait、cancel之一；'
                'kind为bus、train、number或空字符串；target必须是用户本次明确指定的标识，保留字母与前导零；'
                'direction是用户指定的方向，没有则留空。不补全没有提供的目标。只输出JSON。用户：'+text)
        r=self.generate(prompt,max_tokens=100,raw_prompt=True)
        match=re.search(r'\{.*\}',r['text'],re.S)
        if not match or r['finish_reason']!='stop':return dict(action='ask',source='native_model_unparsed')
        intent=json.loads(match[0])
        if intent.get('action')=='wait':
            from .tasks import identifiers
            target=str(intent.get('target','')).upper()
            # Model interpretation cannot invent or truncate a user's identifier.
            if target not in identifiers(text):
                intent.update(target='',needs_clarification=True,source='native_target_not_in_utterance')
        return intent

    def cancel(self):
        if self.handle:self.lib.prts_vlm_cancel(self.handle)

    def close(self):
        self.cancel()
        with self.lock:
            if self.handle:self.lib.prts_vlm_destroy(self.handle);self.handle=None
