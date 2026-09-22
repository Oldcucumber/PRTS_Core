"""Export actual semantic logits and check ONNX Runtime CPU on real images."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch
from prts_core.models import Perception

class Logits(torch.nn.Module):
    def __init__(self,model,kind):super().__init__();self.model=model;self.kind=kind
    def forward(self,x):
        if self.kind=='segformer':return self.model(pixel_values=x).logits
        if self.kind=='mask2former':
            result=self.model(pixel_values=x)
            classes=result.class_queries_logits.softmax(-1)[...,:-1]
            masks=torch.nn.functional.interpolate(result.masks_queries_logits,size=(180,320),mode='bilinear',align_corners=False).sigmoid()
            scores=torch.einsum('bqc,bqhw->bchw',classes,masks)
            return scores/scores.sum(1,keepdim=True).clamp_min(1e-8)
        y=self.model(x)
        return y[0] if isinstance(y,(tuple,list)) else y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='yolo26s-sem.pt')
    ap.add_argument('--width',type=int,default=1024);ap.add_argument('--height',type=int,default=576)
    ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/onnx'))
    ap.add_argument('--images',nargs='+',required=True);args=ap.parse_args()
    torch.set_num_threads(6);torch.manual_seed(0);args.output.mkdir(parents=True,exist_ok=True)
    m=Perception(segmentation=args.model,device='cpu',square=False,long_side=args.width)
    model=Logits(m.seg,m.seg_kind).eval();shape=(1,3,args.height,args.width)
    destination=args.output/'semantic.onnx'
    x=torch.zeros(shape)
    torch.onnx.export(model,x,str(destination),opset_version=17,dynamo=False,input_names=['rgb'],output_names=['logits'])
    import onnx,onnxruntime as ort
    graph=onnx.load(str(destination));grid_casts=0
    if m.seg_kind=='mask2former':
        # Legacy exporter promotes sampling grids to double although the
        # PyTorch CPU forward uses float32. ORT CPU requires float32 grids.
        nodes=[]
        for node in graph.graph.node:
            if node.op_type=='GridSample':
                original=node.input[1];casted=node.name+'/float32_grid'
                nodes.append(onnx.helper.make_node('Cast',[original],[casted],to=onnx.TensorProto.FLOAT))
                node.input[1]=casted;grid_casts+=1
            nodes.append(node)
        del graph.graph.node[:];graph.graph.node.extend(nodes);onnx.save(graph,str(destination))
    onnx.checker.check_model(graph);del graph
    opts=ort.SessionOptions();opts.intra_op_num_threads=6;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(destination),opts,providers=['CPUExecutionProvider'])
    rows=[]
    for image in args.images:
        bgr=cv2.imread(image)
        if bgr is None:raise FileNotFoundError(image)
        rgb=cv2.cvtColor(cv2.resize(bgr,(args.width,args.height)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if m.seg_kind!='yolo_semantic':rgb=(rgb-m.mean)/m.std
        x=np.ascontiguousarray(rgb.transpose(2,0,1)[None])
        with torch.inference_mode():reference=model(torch.from_numpy(x)).numpy()
        start=time.perf_counter();actual=session.run(None,{'rgb':x})[0];duration=(time.perf_counter()-start)*1000
        row={'image':image,'onnx_cpu_ms':duration,'max_abs_error':float(np.max(np.abs(reference-actual))),
             'mean_abs_error':float(np.mean(np.abs(reference-actual))),
             'mask_agreement':float((reference.argmax(1)==actual.argmax(1)).mean())}
        rows.append(row);print(json.dumps(row),flush=True)
    meta={'model':args.model,'format':'ONNX','opset':17,'input_shape':list(shape),'input_name':'rgb',
          'input':'RGB float32 NCHW, range 0..1, direct aspect-preserving resize to fixed input shape',
          'mean':m.mean.tolist() if m.seg_kind!='yolo_semantic' else None,
          'std':m.std.tolist() if m.seg_kind!='yolo_semantic' else None,'labels':m.labels,
          'output':'NCHW class probabilities at 180x320' if m.seg_kind=='mask2former' else 'NCHW class logits',
          'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'bytes':destination.stat().st_size,
          'providers':session.get_providers(),'grid_coordinate_float32_casts':grid_casts,
          'validation':rows,'passed':all(r['mask_agreement']>=.995 and r['max_abs_error']<.005 for r in rows),
          'coreml_status':'not_converted_or_run; no macOS/Xcode in this environment'}
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    assert meta['passed'],meta

if __name__=='__main__':main()
