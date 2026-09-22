"""Fixed numeric image-operation gate; real-model output gate is separate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from prts_core.native_image import NativeImage
from prts_core.onnx_perception import letterbox


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    library=ROOT/'outputs/native-toolchain/prts_image.dll';native=NativeImage(library)
    rng=np.random.default_rng(63059);pre=[];decode=[]
    for w,h in [(432,324),(181,320),(200,97),(160,90)]:
        rgb=rng.integers(0,256,(h,w,3),np.uint8)
        for tw,th in [(640,384),(1024,576)]:
            expected,crop=letterbox(rgb,tw,th);expected=expected.transpose(2,0,1)[None].astype(np.float32)/255
            actual,actual_crop=native.letterbox(rgb,tw,th)
            pixel_error=float(np.abs(actual-expected).max()*255)
            pre.append(dict(source=[w,h],target=[tw,th],crop_matches=actual_crop==crop,max_uint8_error=pixel_error,
                            passed=actual_crop==crop and pixel_error<=1.0001))
            scores=rng.uniform(0,1,(65,180,320)).astype(np.float32)
            gw,gh=round(w*min(1,320/max(h,w))),round(h*min(1,320/max(h,w)))
            ids=[11,15];actual=native.decode(scores,crop,tw,th,gw,gh,ids)
            x,y,rw,rh=crop;xa,ya,xb,yb=round(x*320/tw),round(y*180/th),round((x+rw)*320/tw),round((y+rh)*180/th)
            resized=cv2.resize(scores[:,ya:yb,xa:xb].transpose(1,2,0),(gw,gh),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
            probs=resized/np.maximum(resized.sum(0,keepdims=True),1e-8);classes=probs.argmax(0).astype(np.uint8)
            walk=probs[ids].sum(0);mask=np.isin(classes,ids)&(walk>=.5)
            agreement=float((classes==actual['classes']).mean());error=float(np.max(np.abs(walk-actual['sidewalk'])))
            decode.append(dict(source=[w,h],target=[tw,th],class_agreement=agreement,walk_probability_max_error=error,
                               mask_agreement=float((mask==actual['walkable']).mean()),passed=agreement>=.99999 and error<1e-6 and np.array_equal(mask,actual['walkable'])))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [library,ROOT/'native/portable/prts_image.cpp']}
    report=dict(passed=all(r['passed'] for r in pre+decode),preprocess=pre,decode=decode,sha256=hashes,
                scope='Constructed RGB/probability numeric parity; one-level uint8 resampling difference allowed. Model-level quality is not established by this test.')
    (a.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
