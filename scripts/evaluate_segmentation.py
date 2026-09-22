"""Evaluate fixed models on a held-out SANPO session with released human labels."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from prts_core.models import ROOT, Perception
from prts_core.guidance import Guide, draw
from train_sanpo import SPLITS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',default=['segformer-sidewalk','segformer-sanpo'])
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/heldout-evaluation')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'annotation':'Released SANPO-Real human panoptic labels, not pseudo-labels',
            'split':'heldout','sessions':SPLITS['heldout'],'models':{}}
    for name in args.models:
        model=Perception(segmentation=name);guide=Guide()
        tp=fp=fn=road_fp=road_n=path_true=path_n=0;records=[];panels=[]
        for sid in SPLITS['heldout']:
            folder=ROOT/'outputs/data/sanpo'/sid
            manifest=json.loads((folder/'manifest.json').read_text())
            annotation=json.loads((folder/'annotation_type.json').read_text())
            for entry in manifest['frames']:
                if annotation.get(str(int(Path(entry['file']).stem)))!='HUMAN_ANNOTATED':continue
                frame=cv2.imread(str(folder/entry['file']))
                pred=model(frame);guidance=guide(frame,pred,entry['time_s'])
                truth=np.asarray(Image.open(folder/entry['label']))[:,:,0]
                truth=cv2.resize(truth,(pred['classes'].shape[1],pred['classes'].shape[0]),interpolation=cv2.INTER_NEAREST)
                valid=truth!=0;ground=np.isin(truth,[3,6,17])
                guess=guidance['_ground'];wrong=guess&~ground&valid
                tp+=int((guess&ground&valid).sum());fp+=int(wrong.sum());fn+=int((~guess&ground&valid).sum())
                road_n+=int((truth==1).sum());road_fp+=int((guess&(truth==1)).sum())
                h,w=truth.shape;frame_valid=frame_true=0
                for x,y in guidance['path']:
                    iy=min(h-1,int(y*h));ix=min(w-1,int(x*w))
                    if valid[iy,ix]:frame_valid+=1;frame_true+=int(ground[iy,ix])
                path_n+=frame_valid;path_true+=frame_true
                records.append(dict(file=entry['file'],time_s=entry['time_s'],status=guidance['status'],
                    direction=guidance['direction'],path_coverage=guidance['path_coverage'],
                    path_valid_samples=frame_valid,path_ground_samples=frame_true,
                    detector_path_shift=guidance['detector_path_shift']))
                if len(records) in (1,5,9,13,17,21):
                    view=draw(frame,guidance)
                    bad=cv2.resize(wrong.astype('uint8'),(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                    view[bad]=(view[bad]*.4+np.array([30,20,255])*.6).astype('uint8')
                    view=cv2.resize(view,(384,288));panels.append(view)
        result=dict(frames=len(records),ground_iou=tp/max(1,tp+fp+fn),ground_precision=tp/max(1,tp+fp),
                    ground_recall=tp/max(1,tp+fn),road_as_ground=road_fp/max(1,road_n),
                    path_samples_on_labelled_ground=path_true/max(1,path_n),path_valid_samples=path_n,
                    states={s:sum(r['status']==s for r in records) for s in {r['status'] for r in records}},
                    model_sha256=hashlib.sha256(next(p for p in [ROOT/'models'/name/'model.safetensors',ROOT/'models'/name/'pytorch_model.bin'] if p.exists()).read_bytes()).hexdigest(),records=records)
        report['models'][name]=result
        if panels:cv2.imwrite(str(args.output/(name+'.jpg')),np.concatenate(panels,axis=1))
        print(name,json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')


if __name__=='__main__':main()
