"""Expose ADE20K winning-class confidence without changing learned model weights."""
import json
from pathlib import Path
import onnx
from onnx import helper,TensorProto
for profile in ['fast','quality']:
 p=Path('web/models')/f'floor-{profile}.onnx';m=onnx.load(p)
 if not any(o.name=='semantic_confidence' for o in m.graph.output):
  n=next(n for n in m.graph.node if n.op_type=='Gather' and 'floor_probability' in n.output)
  m.graph.node.append(helper.make_node('ReduceMax',[n.input[0]],['semantic_confidence'],axes=[1],keepdims=0))
  dims=[d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
  m.graph.output.append(helper.make_tensor_value_info('semantic_confidence',TensorProto.FLOAT,dims))
  onnx.checker.check_model(m);onnx.save(m,p)
print('Added confidence output; original outputs and learned weights preserved')
