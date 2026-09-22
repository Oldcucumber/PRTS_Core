"""Local CPU reference using the official llama.cpp server, never a cloud API."""
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import cv2
import requests


class ServerVLM:
    supports_json_schema=True
    def __init__(self,executable,language,projector,log_path,image_tokens=512,sampling='greedy'):
        self.model_name=Path(language).stem+'+official-llama-b10964'
        self.sampling=sampling
        self.lock=threading.Lock();self.response=None;self.cancelled=threading.Event()
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        self.url=f'http://127.0.0.1:{port}'
        Path(log_path).parent.mkdir(parents=True,exist_ok=True)
        self.log=open(log_path,'w',encoding='utf-8')
        command=[str(Path(executable).resolve()),'-m',str(Path(language).resolve()),'--mmproj',str(Path(projector).resolve()),
                 '--host','127.0.0.1','--port',str(port),'-c','4096','-np','1','-t','4','-tb','6','-ngl','0',
                 '--no-mmproj-offload','--image-max-tokens',str(image_tokens),'--reasoning','off','--no-webui']
        self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if self.process.poll() is not None:raise RuntimeError('Local model server failed; inspect '+str(log_path))
            try:
                if requests.get(self.url+'/health',timeout=1).ok:return
            except requests.RequestException:pass
            time.sleep(.1)
        self.close();raise TimeoutError('Local model server did not become ready')

    def generate(self,question,bgr=None,context='',max_tokens=180,raw_prompt=False,on_delta=None,response_schema=None):
        with self.lock:
            self.cancelled.clear();content=[];start=time.perf_counter();pieces=[];reason='length'
            if bgr is not None:
                ok,encoded=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
                if not ok:raise ValueError('Cannot encode camera image')
                content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(encoded).decode())))
            prompt=question if raw_prompt else '你是PRTS视觉助手。根据可见证据简洁回答，不猜测通行许可。\n'+context+'\n用户：'+question
            content.append(dict(type='text',text=prompt))
            payload=dict(messages=[dict(role='user',content=content)],stream=True,max_tokens=max_tokens,temperature=0,
                         chat_template_kwargs=dict(enable_thinking=False),cache_prompt=False)
            if self.sampling=='recommended':
                payload.update(temperature=.7,top_p=.8,top_k=20,min_p=0.,presence_penalty=1.5,repeat_penalty=1.,seed=42)
            if response_schema:payload['response_format']=dict(type='json_object',schema=response_schema)
            try:
                with requests.post(self.url+'/v1/chat/completions',json=payload,stream=True,timeout=(5,120)) as response:
                    self.response=response;response.raise_for_status()
                    for line in response.iter_lines(chunk_size=512):
                        if self.cancelled.is_set():reason='cancelled';break
                        if not line.startswith(b'data: '):continue
                        data=line[6:]
                        if data==b'[DONE]':break
                        choice=json.loads(data)['choices'][0]
                        value=choice.get('delta',{}).get('content','') or ''
                        pieces.append(value)
                        if value and on_delta:on_delta(value)
                        if choice.get('finish_reason'):reason=choice['finish_reason']
            except (requests.RequestException,AttributeError):
                if not self.cancelled.is_set():raise
                reason='cancelled'
            finally:self.response=None
            return dict(text=''.join(pieces).strip(),finish_reason=reason,processing_ms=(time.perf_counter()-start)*1000,
                        model=self.model_name,sampling=self.sampling)

    def intent(self,text):
        from .native_vlm import NativeVLM
        return NativeVLM.intent(self,text)

    def cancel(self):
        self.cancelled.set()
        # requests.close() can wait for an in-progress read. The capture/state
        # thread must return immediately; generation closes at the next SSE chunk.

    def close(self):
        self.cancel()
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()
