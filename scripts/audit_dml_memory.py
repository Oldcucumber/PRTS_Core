"""Record default DirectML adapter memory separately from other DXGI adapters."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import psutil
from prts_core.onnx_perception import ONNXPerception
from prts_core.windows_memory import WindowsGPUMemory


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    gpu=WindowsGPUMemory(ROOT/'outputs/native-toolchain/prts_gpu_memory.dll')
    gpu.lib.prts_gpu_memory_for_adapter.argtypes=[C.c_uint32,C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
    gpu.lib.prts_gpu_memory_for_adapter.restype=C.c_int32
    rows=[]
    def sample(stage):
        entries=[]
        for i in range(8):
            local=C.c_uint64();nonlocal_=C.c_uint64()
            status=gpu.lib.prts_gpu_memory_for_adapter(i,C.byref(local),C.byref(nonlocal_))
            if status<0:break
            entries.append(dict(index=i,local=local.value,nonlocal_=nonlocal_.value))
        info=psutil.Process().memory_info()
        row=dict(stage=stage,rss=info.rss,peak_working_set=info.peak_wset,adapters=entries)
        rows.append(row);print(json.dumps(row),flush=True)
    sample('before_load')
    model=ONNXPerception(a.model,ROOT/'outputs/stage2/perception/onnx/yolo11n-rect',
                         low_memory=True,providers=['DmlExecutionProvider'])
    sample('after_load')
    input_manifest=next((ROOT/'outputs/stage2/perception/data').glob('dev-*/manifest.json'))
    record=json.loads(input_manifest.read_text())['frames'][0]
    image=cv2.imread(str(input_manifest.parent/record['file']))
    for i in range(3):
        start=time.perf_counter();result=model(image)
        sample('after_frame_'+str(i));rows[-1]['wall_ms']=(time.perf_counter()-start)*1000
    a.output.write_text(json.dumps(dict(model=a.model,provider=model.seg.get_providers(),device_id=0,rows=rows,
        rule='Measure DXGI adapter index 0, matching the explicitly selected DML device. Do not sum inactive adapters.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
