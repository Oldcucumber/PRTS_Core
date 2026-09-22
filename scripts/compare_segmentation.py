"""Same-frame pretrained model comparison on explicitly selected development images."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from prts_core.models import Perception
from prts_core.guidance import Guide,draw

p=argparse.ArgumentParser();p.add_argument('--model',default='segformer-cityscapes');p.add_argument('--count',type=int,default=16)
p.add_argument('--size',type=int,default=1024);p.add_argument('--square',action='store_true');p.add_argument('--tag',default='')
p.add_argument('--data',type=Path,default=Path('outputs/data/sanpo'));args=p.parse_args()
m=Perception(segmentation=args.model,long_side=args.size,square=args.square);out=Path('outputs/seg-comparison')/(args.model+args.tag);out.mkdir(parents=True,exist_ok=True)
records=[];tiles=[]
for i,d in enumerate(json.loads((args.data/'discovery.json').read_text())[:args.count]):
 f=cv2.imread(str(args.data/d['session']/'preview.png'));a=m(f);g=Guide()(f,a,0)
 view=draw(f,g);cv2.imwrite(str(out/(d['session']+'.jpg')),view)
 tile=cv2.resize(view,(384,216));cv2.putText(tile,str(i),(5,208),0,.7,(0,0,255),2);tiles.append(tile)
 np.savez_compressed(out/(d['session']+'.npz'),sidewalk=a['sidewalk'],classes=a['classes'],sidewalk_class=a['sidewalk_class'])
 records.append(dict(session=d['session'],**{k:v for k,v in g.items() if not k.startswith('_')},**a['timings']))
 print(i,g['status'],g['direction'],round(g['ground_fraction'],3),flush=True)
while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(out/'contact.jpg'),np.concatenate([np.concatenate(tiles[i:i+4],axis=1) for i in range(0,len(tiles),4)]))
(out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
