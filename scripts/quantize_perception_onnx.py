"""Optional INT8 MatMul weights: measure fidelity before proposing deployment."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic,QuantType

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path('outputs/stage2/perception/onnx/mask2former-1024'))
    ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/onnx/mask2former-int8'))
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    model=a.output/'semantic.onnx'
    quantize_dynamic(str(a.source/'semantic.onnx'),str(model),op_types_to_quantize=['MatMul','Gemm'],
                     weight_type=QuantType.QInt8,per_channel=False,extra_options={'MatMulConstBOnly':True})
    meta=json.loads((a.source/'manifest.json').read_text());meta['quantization']='dynamic INT8 MatMul/Gemm weights; convolution remains FP32'
    meta['float32_source_sha256']=meta['sha256'];meta['sha256']=hashlib.sha256(model.read_bytes()).hexdigest();meta['bytes']=model.stat().st_size
    meta['passed']=False;meta['validation']=[];meta['validation_status']='awaiting_development_fidelity_check'
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2));print(model.stat().st_size,flush=True)

if __name__=='__main__':main()
