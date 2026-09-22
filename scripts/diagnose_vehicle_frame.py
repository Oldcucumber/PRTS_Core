"""Reproduce a streaming evidence failure on the exact cached camera pixels."""
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from prts_core.local_backend import LocalBackend
from prts_core.interfaces import VideoFrame

out=ROOT/'outputs/stage2/streaming/evidence-reproduction';out.mkdir(parents=True,exist_ok=True)
log=(out/'native.log').open('w');os.dup2(log.fileno(),2)
backend=LocalBackend(ROOT/'models',asr_variant='sensevoice',prewarm_audio=True,
    native_library=ROOT/'outputs/prts-native-build/libprts_vlm.dll',
    onnx_semantic_dir=ROOT/'outputs/stage2/perception/onnx/mask2former-1024-fp16-v2',
    onnx_detector_dir=ROOT/'outputs/stage2/perception/onnx/yolo11n-rect',onnx_provider='DmlExecutionProvider')
backend._load_brain()
try:
    for i in range(18,37):
        frame=cv2.imread(str(ROOT/f'outputs/stage2/streaming/bus-dml-v3/input-frames/{i:06}.png'))
        try:
            result=backend.collect_frame_evidence(VideoFrame(i*.5,i,frame),dict(kind='bus'))
            (out/f'{i:06}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(i,[(e['text'],e.get('route_verified')) for e in result['ocr']],flush=True)
        except Exception:
            (out/f'{i:06}-traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');print(i,'failed',flush=True);break
finally:backend._brain.close()
