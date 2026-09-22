"""Candidate-only Q8_0 matrix quantization with the pinned runtime's GGUF tools.

Keep convolution, position embeddings, bias/norm and non-block-aligned matrices
at their original precision. Original F16 projector is never overwritten.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/vendor/MiniCPM-V-Apps/llama.cpp-omni/gguf-py'))
import gguf
import numpy as np


def main():
    folder=ROOT/'models/minicpm-v-4.6-gguf'
    source=folder/'mmproj-model-f16.gguf'
    target=folder/'mmproj-matrix-q8_0.gguf'
    if target.exists():raise FileExistsError('Preserve the previous projector candidate')
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    if original!='ca931d861d0801d9003e50697cd764721a334107c0e0415a51168ee1938462de':
        raise ValueError('Unexpected source projector hash')
    reader=gguf.GGUFReader(source)
    writer=gguf.GGUFWriter(target,'clip',use_temp_file=True)
    for key,field in reader.fields.items():
        if key.startswith('GGUF.') or key in ('general.architecture','general.file_type','general.quantization_version'):continue
        writer.add_key_value(key,field.contents(),field.types[0],field.types[1] if len(field.types)>1 else None)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    rows=[]
    for tensor in reader.tensors:
        data=tensor.data;old=tensor.tensor_type;kind=old
        selected=(old==gguf.GGMLQuantizationType.F16 and len(data.shape)==2
                  and tensor.name.endswith('.weight') and data.shape[-1]%32==0
                  and 'position' not in tensor.name and 'norm' not in tensor.name)
        if selected:
            kind=gguf.GGMLQuantizationType.Q8_0
            data=gguf.quantize(data.astype(np.float32),kind)
        writer.add_tensor(tensor.name,data,raw_dtype=kind)
        rows.append(dict(name=tensor.name,shape=tensor.shape.tolist(),old_type=old.name,new_type=kind.name))
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    check=gguf.GGUFReader(target)
    if [(t.name,list(t.shape)) for t in check.tensors]!=[(t.name,list(t.shape)) for t in reader.tensors]:
        raise ValueError('Tensor names or logical shapes changed')
    report=dict(source=source.name,source_sha256=original,file=target.name,bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),status='Unvalidated candidate; do not adopt before task and memory gates',
        runtime_revision='13401aa8480d68a725e131cc4dbe5eda1f996a7f',
        recipe='F16 aligned 2D weight matrices to Q8_0; retain all other original types',tensors=rows)
    (folder/'mmproj-matrix-q8_0-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tensors'}))


if __name__=='__main__':main()
