"""Export Depth Anything V2 Small for browser relative inverse-depth inference."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForDepthEstimation

MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
SIZE = 252  # multiple of the ViT's 14-pixel patch size


class DepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL, attn_implementation='eager').eval()

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


def main():
    torch.set_num_threads(4)
    target = Path('web/models')
    target.mkdir(exist_ok=True)
    model = DepthModel().eval()
    sample = torch.zeros((1,3,SIZE,SIZE),dtype=torch.float32)
    path = target/'depth-small.onnx'
    torch.onnx.export(model,sample,str(path),input_names=['pixel_values'],output_names=['relative_depth'],opset_version=17,dynamo=False)
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    cap=cv2.VideoCapture('VID20260919182406.mp4')
    report={'model':MODEL,'input_size':SIZE,'bytes':path.stat().st_size,'meaning':'relative inverse depth; larger means nearer; not meters','frames':[]}
    for index in [0,90,180,270]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
        if not ok:raise RuntimeError('Cannot read test video')
        h,w=frame.shape[:2];scale=min(SIZE/w,SIZE/h);rw,rh=round(w*scale),round(h*scale)
        rgb=np.full((SIZE,SIZE,3),[124,116,104],np.uint8)
        x,y=(SIZE-rw)//2,(SIZE-rh)//2
        rgb[y:y+rh,x:x+rw]=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(rw,rh))
        normalized=(rgb.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        data=normalized.transpose(2,0,1)[None].copy()
        t=time.perf_counter();actual=session.run(None,{'pixel_values':data})[0];ms=(time.perf_counter()-t)*1000
        with torch.inference_mode():expected=model(torch.from_numpy(data)).numpy()
        error=float(np.max(np.abs(actual-expected)));scale=float(np.max(np.abs(expected)))
        assert error/max(scale,1e-6)<1e-4,(error,scale)
        assert np.isfinite(actual).all() and float(np.ptp(actual))>0
        item={'frame':index,'max_absolute_error':error,'relative_error':error/max(scale,1e-6),'inference_ms':ms,'min':float(actual.min()),'max':float(actual.max())}
        report['frames'].append(item);print(item,flush=True)
    cap.release()
    assert path.stat().st_size < 100*1024*1024,'Exceeds GitHub per-file limit'
    (target/'depth-export-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Exported',report['bytes'],'bytes',flush=True)


if __name__=='__main__':main()
