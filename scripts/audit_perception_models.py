"""Audit the unused Swin pool layer and exercise dense semantic model loading."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch
from prts_core.models import Perception
from prts_core.semantics import semantic_view

def main():
    out=Path('outputs/stage2/perception/model-audit');out.mkdir(parents=True,exist_ok=True)
    frame=cv2.imread('outputs/data/sanpo/28tdwxgz-zPU06lpeDY3OPWTFxZyBv2c/frames/000072.png')
    if frame is None:raise FileNotFoundError('audit frame')
    m=Perception(segmentation='mask2former-mapillary',long_side=512,square=False)
    norm=m.seg.model.pixel_level_module.encoder.swin.layernorm
    calls=[]
    handle=norm.register_forward_hook(lambda *args:calls.append(True))
    with torch.inference_mode():
        x=torch.randn(1,3,288,512,device='cuda')
        a=m.seg(pixel_values=x)
        norm.weight.fill_(7);norm.bias.fill_(-11)
        b=m.seg(pixel_values=x)
    audit={'missing_layernorm_forward_calls':len(calls),
           'class_logits_max_abs_delta':float((a.class_queries_logits-b.class_queries_logits).abs().max()),
           'mask_logits_max_abs_delta':float((a.masks_queries_logits-b.masks_queries_logits).abs().max()),
           'note':'The missing global Swin layernorm is computed, but does not affect segmentation logits: perturbing its weight to 7 and bias to -11 gives exactly zero delta. Feature maps use the stage-specific norms.'}
    handle.remove()
    assert audit['class_logits_max_abs_delta']==0 and audit['mask_logits_max_abs_delta']==0,audit
    p=m(frame);cv2.imwrite(str(out/'mask2former.jpg'),semantic_view(frame,p))
    audit['mask2former_timings']=p['timings']
    del a,b,m;torch.cuda.empty_cache()
    m=Perception(segmentation='yolo26l-sem.pt',long_side=512,square=False)
    p=m(frame);cv2.imwrite(str(out/'yolo26l.jpg'),semantic_view(frame,p))
    audit['yolo26l_timings']=p['timings'];audit['yolo26l_labels']=p['labels']
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
