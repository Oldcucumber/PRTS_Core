"""Run the exported models through C ABI and Python ORT with identical tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import onnxruntime as ort
from prts_core.native_onnx import NativeONNX
from prts_core.onnx_perception import letterbox


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--semantic',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    log=(a.output/'runtime-stderr.log').open('wb');os.dup2(log.fileno(),2)
    source=ROOT/'outputs/stage2/bus-generalization/data/commons-nwfb-101x.JPG'
    image=cv2.imread(str(source));bridge=ROOT/'outputs/native-toolchain/prts_onnx.dll'
    runtime=Path(ort.__file__).parent/'capi/onnxruntime.dll'
    folders=[ROOT/'outputs/stage2/perception/onnx/yolo11n-rect']
    if a.semantic:folders.append(ROOT/'outputs/stage2/perception/onnx/mask2former-1024-fp16-v2')
    results=[]
    for folder in folders:
        meta=json.loads((folder/'manifest.json').read_text());model=folder/('semantic.onnx' if 'mask2former' in folder.name else 'detector.onnx')
        ih,iw=meta['input_shape'][2:];canvas,_=letterbox(image,iw,ih)
        rgb=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if meta.get('mean') is not None:rgb=(rgb-np.asarray(meta['mean'],np.float32))/np.asarray(meta['std'],np.float32)
        x=np.ascontiguousarray(rgb.transpose(2,0,1)[None]);native=NativeONNX(bridge,runtime,model)
        native_value=native.run(x);native.close()
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
        options.add_session_config_entry('session.disable_prepacking','1')
        session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
        expected=session.run(None,{'rgb':x})[0];del session
        error=float(np.max(np.abs(native_value-expected)))
        results.append(dict(model=str(model.relative_to(ROOT)),shape=list(native_value.shape),
                            max_absolute_error=error,exact=np.array_equal(native_value,expected),passed=error<1e-6))
        np.savez_compressed(a.output/(folder.name+'.npz'),input=x,native=native_value,reference=expected)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [bridge,source,ROOT/'native/portable/prts_onnx.cpp']}
    report=dict(passed=all(r['passed'] for r in results),results=results,ort_version=ort.__version__,sha256=hashes,
                scope='Windows CPU C ABI parity; not CoreML provider or Swift compilation.')
    (a.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
