"""Export fixed-shape YOLO11n raw predictions; compare actual ORT CPU outputs."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch,onnx,onnxruntime as ort
from prts_core.models import ROOT
from prts_core.onnx_perception import letterbox
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--images',nargs='+',required=True)
    ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/onnx/yolo11n'))
    ap.add_argument('--height',type=int,default=384);ap.add_argument('--width',type=int,default=640)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6)
    yolo=YOLO(str(ROOT/'models/yolo11n.pt'));model=yolo.model.float().eval()
    head=model.model[-1];head.export=True;head.format='onnx';head.dynamic=False
    path=a.output/'detector.onnx'
    torch.onnx.export(model,torch.zeros(1,3,a.height,a.width),str(path),opset_version=17,dynamo=False,input_names=['rgb'],output_names=['predictions'])
    onnx.checker.check_model(onnx.load(str(path)))
    opts=ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider']);rows=[]
    for p in a.images:
        canvas,_=letterbox(cv2.imread(p),a.width,a.height)
        x=np.ascontiguousarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1)[None].astype(np.float32)/255)
        with torch.inference_mode():expected=model(torch.from_numpy(x)).numpy()
        start=time.perf_counter();actual=session.run(None,{'rgb':x})[0]
        rows.append({'image':p,'cpu_ms':(time.perf_counter()-start)*1000,'max_abs_error':float(np.abs(expected-actual).max()),
                     'max_box_error_normalized':float((np.abs(expected[:,:4]-actual[:,:4])/np.asarray([a.width,a.height,a.width,a.height])[None,:,None]).max()),
                     'max_class_score_error':float(np.abs(expected[:,4:]-actual[:,4:]).max()),
                     'threshold_mask_equal':bool(np.array_equal(expected[:,4:].max(1)>=.25,actual[:,4:].max(1)>=.25)),
                     'mean_abs_error':float(np.abs(expected-actual).mean())})
    meta={'model':'yolo11n.pt','input_shape':[1,3,a.height,a.width],'labels':yolo.names,'output':'N x 84 x anchors, xywh pixels plus 80 class scores',
          'input':'RGB float32 NCHW 0..1, letterbox 114','providers':session.get_providers(),'validation':rows,
          'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'validation_limits':{'normalized_box_error':.0001,'class_score_error':.0001,'threshold_mask_equal':True},
          'passed':all(r['max_box_error_normalized']<.0001 and r['max_class_score_error']<.0001 and r['threshold_mask_equal'] for r in rows)}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta));assert meta['passed']

if __name__=='__main__':main()
