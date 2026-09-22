"""Small supervised domain adaptation using ONLY SANPO's released segmentation labels."""
import argparse,json,random,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prts_core.models import ROOT
import cv2,numpy as np,torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

SPLITS={
 'train':['28tdwxgz-zPU06lpeDY3OPWTFxZyBv2c','3fNdc-1R7QcLCPHG7Zn9oQXrPqKQbuys','5z7n1PIsHGyQmO64hAlMynF3Cdj0rBeu','8GbY4HVFE794yBRg8-wMqeDi_uLHjQPX','ALYuoq-pWdqtJ-Jm9R8ceVzhsXYawzfi'],
 'development':['AMWsWL1lxlW0NelUU8w8yL4psb1R1vCd'],
 'heldout':['AdtGuq73TlhK5miUkWcUO9BoKdnmV-8j']}


def dataset(root,sessions,size=512):
    samples=[]
    for sid in sessions:
        folder=root/sid;manifest=json.loads((folder/'manifest.json').read_text())
        annotation=json.loads((folder/'annotation_type.json').read_text())
        for f in manifest['frames']:
            if not f['label']:continue
            if annotation.get(str(int(Path(f['file']).stem)))!='HUMAN_ANNOTATED':continue
            image=cv2.cvtColor(cv2.imread(str(folder/f['file'])),cv2.COLOR_BGR2RGB)
            label=np.asarray(Image.open(folder/f['label']))[:,:,0]
            image=cv2.resize(image,(size,size));label=cv2.resize(label,(size,size),interpolation=cv2.INTER_NEAREST).copy()
            label[label==0]=255
            samples.append((image,label))
    return samples


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=400);ap.add_argument('--seed',type=int,default=73)
    args=ap.parse_args();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    data=ROOT/'outputs/data/sanpo';out=ROOT/'models/segformer-sanpo';out.mkdir(parents=True,exist_ok=True)
    train=dataset(data,SPLITS['train']);dev=dataset(data,SPLITS['development'])
    if not train or not dev:raise RuntimeError('Need downloaded public labels for training and distinct development sessions')
    labels=json.loads((data/'labelmap.json').read_text());id2label={v:k for k,v in labels.items()}
    model=SegformerForSemanticSegmentation.from_pretrained(str(ROOT/'models/segformer-cityscapes'),local_files_only=True,
            num_labels=31,id2label=id2label,label2id=labels,ignore_mismatched_sizes=True).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=6e-5,weight_decay=.01)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    def batch(samples,augment=False):
        images=[];masks=[]
        for image,label in samples:
            if augment and random.random()<.5:image=image[:,::-1];label=label[:,::-1]
            x=image.copy().astype(np.float32)/255
            if augment:x=np.clip(x*random.uniform(.8,1.2)+random.uniform(-.08,.08),0,1)
            images.append(x);masks.append(label.copy())
        x=torch.from_numpy(np.stack(images).transpose(0,3,1,2)).cuda();y=torch.from_numpy(np.stack(masks).astype(np.int64)).cuda()
        return (x-mean)/std,y
    @torch.inference_mode()
    def evaluate():
        model.eval();inter=union=road_fp=road_n=0;tp=fp=fn=0
        for sample in dev:
            x,y=batch([sample]);logits=model(pixel_values=x).logits
            pred=torch.nn.functional.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
            valid=y!=255;truth=(y==3)|(y==6)|(y==17);guess=(pred==3)|(pred==6)|(pred==17)
            tp+=int((truth&guess&valid).sum());fp+=int((~truth&guess&valid).sum());fn+=int((truth&~guess&valid).sum())
            road_n+=int((y==1).sum());road_fp+=int(((y==1)&guess).sum())
        return dict(ground_iou=tp/max(1,tp+fp+fn),ground_precision=tp/max(1,tp+fp),ground_recall=tp/max(1,tp+fn),road_as_ground=road_fp/max(1,road_n))
    best=-1;history=[];started=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train();x,y=batch(random.choices(train,k=4),True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(pixel_values=x,labels=y).loss
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step%50==0 or step==args.steps:
            metrics=evaluate();record=dict(step=step,loss=float(loss.item()),seconds=time.perf_counter()-started,**metrics)
            history.append(record);print(json.dumps(record),flush=True)
            # Reject checkpoints that absorb road into walkable ground on development.
            # Held-out labels never influence selection.
            if metrics['road_as_ground']<=.02 and metrics['ground_iou']>best:
                best=metrics['ground_iou'];model.save_pretrained(out);(out/'selected.json').write_text(json.dumps(record,indent=2))
    report=dict(source='https://github.com/google-research-datasets/sanpo_dataset',license='CC-BY-4.0 data; NVIDIA upstream weight restrictions remain',
                annotation='Only frames marked HUMAN_ANNOTATED by the released SANPO metadata; MACHINE_ANNOTATED excluded',splits=SPLITS,seed=args.seed,steps=args.steps,
                train_frames=len(train),development_frames=len(dev),history=history,heldout_used_during_training=False,
                selection_rule='Maximum development ground IoU subject to road_as_ground <= 0.02',
                eligible_checkpoint_found=best>=0)
    if best<0:raise RuntimeError('No checkpoint met development road error constraint; do not promote these weights')
    (out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    target=ROOT/'outputs/training';target.mkdir(exist_ok=True)
    (target/'sanpo.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
