"""Create a separate FP16 ONNX candidate, preserving FP32 external IO."""
import argparse
import hashlib
import json
from pathlib import Path
import onnx
from onnxconverter_common import float16


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    destination=a.output/'semantic.onnx'
    if destination.exists():raise FileExistsError('Preserve previous candidates')
    source=a.source/'semantic.onnx'
    model=onnx.load(source)
    model=float16.convert_float_to_float16(model,keep_io_types=True)
    # Converter 1.16 changes inferred output types but leaves some existing
    # integer-to-float Cast attributes at FLOAT. Reconcile only declared FP16
    # outputs; preserve intentional FP32 boundaries around blocked operators.
    types={v.name:v.type.tensor_type.elem_type for v in
           list(model.graph.value_info)+list(model.graph.input)+list(model.graph.output)}
    fixed=[]
    for node in model.graph.node:
        if node.op_type=='Cast' and types.get(node.output[0])==onnx.TensorProto.FLOAT16:
            attribute=next(a for a in node.attribute if a.name=='to')
            if attribute.i==onnx.TensorProto.FLOAT:
                attribute.i=onnx.TensorProto.FLOAT16;fixed.append(node.name)
    onnx.checker.check_model(model,full_check=True)
    onnx.save(model,destination)
    meta=json.loads((a.source/'manifest.json').read_text(encoding='utf-8'))
    meta.update(precision='FP16 with FP32 external IO and converter default blocked operators',
                parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                converter='onnxconverter-common 1.16.0',bytes=destination.stat().st_size,
                reconciled_casts=fixed,
                quality_status='Unvalidated candidate; run frozen comparison before adoption')
    (a.output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(bytes=meta['bytes'],sha256=meta['sha256'])))


if __name__=='__main__':main()
