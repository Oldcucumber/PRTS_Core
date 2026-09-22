"""Exercise real native streaming cancellation, Unicode and handle reuse."""
import json
import os
from pathlib import Path
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from prts_core.native_vlm import NativeVLM


def main():
    out=ROOT/'outputs/stage2/native-vlm/contract';out.mkdir(parents=True,exist_ok=True)
    log=(out/'native.log').open('w');os.dup2(log.fileno(),2)
    model=NativeVLM(ROOT/'models',ROOT/'outputs/prts-native-build/libprts_vlm.dll',image_slices=1)
    results={};deltas=[]
    try:
        def delta(text):deltas.append(text);model.cancel()
        results['cancel_after_token']=model.generate('请从一数到一百，每个数字之间用中文逗号。',max_tokens=100,on_delta=delta)
        results['unicode_reuse']=model.generate('请原样输出：你好，PRTS。',max_tokens=30,raw_prompt=True)
        image=cv2.imread(str(ROOT/'outputs/stage2/native-vlm/quantization-v2/route-20.png'))
        started=threading.Event();error=[]
        def run():
            try:
                started.set();results['cancel_during_image']=model.generate('详细描述画面。',image,max_tokens=100)
            except Exception as e:error.append(repr(e))
        worker=threading.Thread(target=run);worker.start();started.wait();time.sleep(.1)
        stamp=time.perf_counter();model.cancel();worker.join(20)
        if worker.is_alive():raise TimeoutError('Native cancellation did not finish')
        results['cancel_to_return_ms']=(time.perf_counter()-stamp)*1000
        results['reuse_after_image_cancel']=model.generate('请回答七加五等于多少。',max_tokens=30)
        passed=(not error and results['cancel_after_token']['finish_reason']=='cancelled' and
                results['cancel_during_image']['finish_reason']=='cancelled' and
                results['unicode_reuse']['finish_reason']=='stop' and '你好' in results['unicode_reuse']['text'] and
                results['reuse_after_image_cancel']['finish_reason']=='stop' and
                all('\ufffd' not in r['text'] for r in results.values() if isinstance(r,dict)))
        results.update(passed=passed,callback_deltas=deltas,errors=error,
            limitation='Visual encoding checks cancellation after the current image evaluation; this is not a hard real-time deadline.')
    finally:model.close()
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
